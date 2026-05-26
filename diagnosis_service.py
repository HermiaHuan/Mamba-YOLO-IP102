from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}
RISK_LABELS = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
    "critical": "极高风险",
}

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


class DiagnosisAPIError(RuntimeError):
    pass


def load_local_env(env_file: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()


def calculate_risk_level(count: int, max_confidence: float, risk_rule: dict | None) -> str:
    if not risk_rule:
        return "low"

    count_thresholds = risk_rule.get("count_thresholds", {})
    confidence_thresholds = risk_rule.get("confidence_thresholds", {})

    for level in ("critical", "high", "medium"):
        if count >= count_thresholds.get(level, 10**9):
            return level
        if max_confidence >= confidence_thresholds.get(level, 2.0):
            return level
    return "low"


def risk_label(level: str) -> str:
    return RISK_LABELS.get(level, RISK_LABELS["low"])


def build_structured_context(
    task_type: str,
    class_stats: list[dict],
    kb_hits: list[dict],
    *,
    environment_note: str = "",
) -> dict:
    return {
        "task_type": task_type,
        "detected_classes": class_stats,
        "knowledge_base_hits": kb_hits,
        "environment_note": environment_note.strip(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise DiagnosisAPIError("LLM 返回中未找到 JSON 对象。")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise DiagnosisAPIError("LLM 返回 JSON 解析失败。") from exc


def call_llm_api(context: dict) -> dict:
    api_key = (os.getenv("AGRI_LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")).strip()
    api_url = os.getenv("AGRI_LLM_API_URL", "https://api.openai.com/v1/chat/completions").strip()
    model_name = os.getenv("AGRI_LLM_MODEL", "gpt-4o-mini").strip()
    timeout = float(os.getenv("AGRI_LLM_TIMEOUT", "20"))
    retries = int(os.getenv("AGRI_LLM_RETRIES", "2"))

    if not api_key:
        raise DiagnosisAPIError("未配置 OPENAI_API_KEY 或 AGRI_LLM_API_KEY。")

    system_prompt = (
        "你是一名农业病虫害诊断助手。"
        "请根据给定的检测统计、农业知识库和田间环境信息，输出严格 JSON。"
        "JSON 键必须只有 diagnosis_conclusion、risk_level、control_advice、precautions。"
        "risk_level 只允许使用 低风险、中风险、高风险、极高风险。"
        "建议应谨慎、可执行，并提醒用户结合人工复核。"
    )
    user_prompt = json.dumps(context, ensure_ascii=False, indent=2)

    payload = {
        "model": model_name,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    response_payload = None
    last_error = None
    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_response = response.read().decode("utf-8")
            try:
                response_payload = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise DiagnosisAPIError("LLM API 返回内容不是合法 JSON。") from exc
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            body = exc.read().decode("utf-8", errors="replace").strip()
            body_hint = f"，响应内容：{body[:300]}" if body else ""
            if exc.code in retryable_statuses and attempt < retries:
                time.sleep(min(2**attempt, 4))
                continue
            raise DiagnosisAPIError(
                f"LLM API 调用失败: HTTP {exc.code} {exc.reason}，"
                f"接口：{api_url}，模型：{model_name}{body_hint}"
            ) from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
                continue
            raise DiagnosisAPIError(
                f"LLM API 调用失败: 网络或网关错误 {exc.reason}，接口：{api_url}，模型：{model_name}"
            ) from exc

    if response_payload is None:
        raise DiagnosisAPIError(f"LLM API 调用失败: {last_error}") from last_error

    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DiagnosisAPIError("LLM API 返回格式不符合预期。") from exc

    return _extract_json_object(content)


def render_local_template(context: dict) -> dict:
    detected_classes = context.get("detected_classes", [])
    kb_lookup = {entry["class_name"]: entry for entry in context.get("knowledge_base_hits", [])}
    task_type = context.get("task_type", "当前任务")
    environment_note = context.get("environment_note", "").strip()

    if not detected_classes:
        return {
            "diagnosis_conclusion": f"当前图片未检出与“{task_type}”对应的有效目标，建议继续保持常规监测。",
            "risk_level": "低风险",
            "control_advice": "保持例行巡检，发现疑似症状或虫体时重新拍摄清晰图像并复检。",
            "precautions": "优先在光照充足、对焦清晰的条件下采图，并结合田间环境持续观察。",
        }

    max_risk = "low"
    conclusion_parts: list[str] = []
    control_parts: list[str] = []
    precaution_parts: list[str] = []

    for item in detected_classes:
        class_name = item["class_name"]
        kb_entry = kb_lookup.get(class_name, {})
        risk_level = item.get("risk_level", "low")
        if RISK_PRIORITY[risk_level] > RISK_PRIORITY[max_risk]:
            max_risk = risk_level

        conclusion_parts.append(
            f"检测到 {item['count']} 个 {class_name}，最大置信度为 {item['max_confidence']:.2f}，"
            f"对应诊断标签为 {item.get('diagnosis_tag', risk_label(risk_level))}。"
        )
        control_parts.append(kb_entry.get("suggested_actions", "建议结合农业技术人员意见实施综合防控。"))
        precaution_parts.append(kb_entry.get("trigger_conditions", "请结合田间环境、气候和作物长势进行复核。"))

    if environment_note:
        precaution_parts.append(f"用户补充的田间信息：{environment_note}")

    return {
        "diagnosis_conclusion": " ".join(conclusion_parts),
        "risk_level": risk_label(max_risk),
        "control_advice": "；".join(dict.fromkeys(control_parts)),
        "precautions": "；".join(dict.fromkeys(precaution_parts)),
    }


def format_diagnosis_text(payload: dict) -> str:
    return (
        f"诊断结论：{payload.get('diagnosis_conclusion', '暂无')}\n\n"
        f"风险等级：{payload.get('risk_level', '低风险')}\n\n"
        f"防治建议：{payload.get('control_advice', '暂无')}\n\n"
        f"注意事项：{payload.get('precautions', '暂无')}"
    )


def generate_diagnosis(context: dict) -> tuple[str, dict, str]:
    try:
        payload = call_llm_api(context)
        source = "llm_api"
    except Exception:
        payload = render_local_template(context)
        source = "local_template"

    required_fields = {
        "diagnosis_conclusion": payload.get("diagnosis_conclusion", ""),
        "risk_level": payload.get("risk_level", "低风险"),
        "control_advice": payload.get("control_advice", ""),
        "precautions": payload.get("precautions", ""),
    }
    return source, required_fields, format_diagnosis_text(required_fields)
