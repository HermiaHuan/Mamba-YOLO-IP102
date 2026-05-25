from __future__ import annotations

import inspect
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path


try:
    import gradio as gr

    GRADIO_IMPORT_ERROR = None
except ImportError as exc:
    gr = None
    GRADIO_IMPORT_ERROR = exc

try:
    from ultralytics import YOLO

    YOLO_IMPORT_ERROR = None
except ImportError as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = exc

from diagnosis_service import (
    build_structured_context,
    calculate_risk_level,
    call_llm_api,
    format_diagnosis_text,
    render_local_template,
    risk_label,
)
from knowledge_base import DISEASE, PEST, KnowledgeBase, load_default_class_names


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_EN_NAME = "Mamba-YOLO IP102"
PROJECT_CN_NAME = "农业害虫智能检测与诊断系统"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "pest102"
DATASET_YAML = DATASET_ROOT / "pest102.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output_dir" / "pest102" / "mambayolo_t_pest102" / "weights" / "best.pt"
RUNTIME_RECORD_DIR = PROJECT_ROOT / "runtime_records"
RUNTIME_RECORD_FILE = RUNTIME_RECORD_DIR / "diagnosis_history.jsonl"
MODEL_SOURCE_LABEL = "Mamba-YOLO-T IP102 训练权重"
TASK_CATEGORY_MAP = {"害虫检测": PEST, "病害检测": DISEASE}
CATEGORY_LABEL_MAP = {"全部": "全部", "害虫": PEST, "病害": DISEASE}
MODEL_CACHE: dict[str, object] = {}
KB_CACHE: dict[tuple[str, ...], KnowledgeBase] = {}

APP_CSS = """
:root {
    --agri-ink: #102218;
    --agri-muted: #3f5145;
    --agri-leaf: #245b3b;
    --agri-moss: #8aa34a;
    --agri-earth: #6f452d;
    --agri-cream: #fbfaf4;
    --agri-panel: #ffffff;
    --agri-soft: #f4faf3;
    --agri-field: #eef7ed;
    --agri-line: rgba(36, 91, 59, 0.24);
    --agri-shadow: 0 18px 45px rgba(32, 54, 39, 0.14);
}

html,
body,
#root,
.gradio-container,
.gradio-container > div {
    background:
        radial-gradient(circle at top left, rgba(138, 163, 74, 0.10), transparent 30%),
        radial-gradient(circle at top right, rgba(36, 91, 59, 0.10), transparent 26%),
        linear-gradient(180deg, #fbfaf3 0%, #f4faf3 48%, #fcfbf6 100%);
    color: var(--agri-ink);
    font-family: "Avenir Next", "SF Pro Display", "PingFang SC", "Microsoft YaHei", sans-serif;
}

.gradio-container {
    max-width: none !important;
    min-height: 100vh !important;
    width: 100% !important;
}

#app-shell {
    box-sizing: border-box;
    margin: 0 auto;
    max-width: 1560px;
    padding: 18px 28px 28px;
    width: 100%;
}

.hero-banner {
    background:
        radial-gradient(circle at 96% 12%, rgba(70, 132, 86, 0.24), transparent 26%),
        linear-gradient(135deg, #deefde 0%, #cbe4cc 54%, #b8d7bb 100%);
    border: 1px solid rgba(36, 91, 59, 0.26);
    border-radius: 28px;
    box-shadow: var(--agri-shadow);
    color: var(--agri-ink);
    overflow: hidden;
    padding: 30px 34px;
    position: relative;
}

.hero-banner::after {
    background:
        radial-gradient(circle, rgba(95, 154, 112, 0.14), transparent 58%);
    content: "";
    height: 240px;
    position: absolute;
    right: -80px;
    top: -80px;
    width: 240px;
}

.hero-kicker {
    color: var(--agri-leaf);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.18em;
    margin-bottom: 10px;
    text-transform: uppercase;
}

.hero-title {
    font-family: "Georgia", "Songti SC", "STSong", serif;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: 0.02em;
    line-height: 1.1;
    margin: 0;
}

.hero-subtitle {
    color: var(--agri-muted);
    font-size: 15px;
    line-height: 1.8;
    margin-top: 16px;
    max-width: 780px;
}

.hero-chip-row,
.quick-card-grid {
    display: grid;
    gap: 12px;
}

.hero-chip-row {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    margin-top: 24px;
}

.hero-chip {
    backdrop-filter: blur(6px);
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(36, 91, 59, 0.20);
    border-radius: 18px;
    padding: 14px 16px;
}

.hero-chip-label {
    color: var(--agri-muted);
    font-size: 12px;
    margin-bottom: 6px;
}

.hero-chip-value {
    color: var(--agri-ink);
    font-size: 18px;
    font-weight: 700;
}

.quick-card-grid {
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    margin: 18px 0 4px;
}

.quick-card {
    background: var(--agri-panel);
    border: 1px solid var(--agri-line);
    border-radius: 22px;
    box-shadow: var(--agri-shadow);
    min-height: 132px;
    padding: 20px 22px;
}

.quick-card-title {
    color: var(--agri-leaf);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 10px;
}

.quick-card-text {
    color: var(--agri-muted);
    font-size: 14px;
    line-height: 1.75;
}

.section-caption {
    color: var(--agri-earth);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.16em;
    margin: 12px 0 4px;
    text-transform: uppercase;
}

.section-title {
    color: var(--agri-ink);
    font-family: "Georgia", "Songti SC", "STSong", serif;
    font-size: 26px;
    margin: 0 0 16px;
}

.panel-card {
    background: var(--agri-panel);
    border: 1px solid var(--agri-line);
    border-radius: 24px;
    box-shadow: var(--agri-shadow);
    padding: 8px !important;
}

.panel-card.compact-card {
    min-height: 100%;
}

.panel-heading {
    color: var(--agri-ink);
    font-size: 18px;
    font-weight: 700;
    margin: 6px 10px 4px;
}

.panel-note {
    color: var(--agri-muted);
    font-size: 13px;
    margin: 0 10px 8px;
}

.gradio-container .gr-button-primary,
.gradio-container .primary-btn {
    background: linear-gradient(135deg, var(--agri-leaf), #3c8652) !important;
    border: none !important;
    border-radius: 16px !important;
    box-shadow: 0 14px 24px rgba(47, 107, 71, 0.22);
    color: #f8f7f1 !important;
    font-weight: 700 !important;
}

.gradio-container .gr-button-primary:hover,
.gradio-container .primary-btn:hover {
    filter: brightness(1.04);
}

.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-input,
.gradio-container .gr-textbox,
.gradio-container .gr-dropdown,
.gradio-container .gr-image,
.gradio-container .gradio-tabitem {
    background: var(--agri-panel) !important;
    border-color: var(--agri-line) !important;
    border-radius: 18px !important;
    color: var(--agri-ink) !important;
}

.gradio-container .tabs {
    border: none !important;
    color: var(--agri-ink) !important;
}

.gradio-container .tab-nav,
.gradio-container .tabitem,
.gradio-container [role="tablist"] {
    background: #ffffff !important;
    border-bottom: 1px solid rgba(36, 91, 59, 0.22) !important;
}

.gradio-container .tab-nav button {
    border-radius: 999px !important;
    background: #ffffff !important;
    border: 1px solid var(--agri-line) !important;
    color: var(--agri-ink) !important;
    font-weight: 700 !important;
    margin-right: 8px;
}

.gradio-container .tab-nav button.selected {
    background: #dfeee1 !important;
    color: #14351f !important;
}

.gradio-container .tab-nav button *,
.gradio-container [role="tab"],
.gradio-container [role="tab"] *,
.gradio-container button[role="tab"],
.gradio-container button[role="tab"] * {
    color: var(--agri-ink) !important;
    opacity: 1 !important;
}

.gradio-container .tab-nav button.selected *,
.gradio-container [role="tab"][aria-selected="true"],
.gradio-container [role="tab"][aria-selected="true"] * {
    color: var(--agri-leaf) !important;
    opacity: 1 !important;
}

.gradio-container table {
    border-collapse: separate;
    border-spacing: 0;
    background: #ffffff !important;
    color: var(--agri-ink) !important;
    font-size: 14px;
    overflow: hidden;
    width: 100%;
}

.gradio-container table thead tr {
    background: #f0f8ef !important;
}

.gradio-container table th,
.gradio-container table td {
    background: #ffffff !important;
    border-bottom: 1px solid rgba(36, 91, 59, 0.16);
    color: var(--agri-ink) !important;
    padding: 10px 12px !important;
}

.gradio-container table th,
.gradio-container table thead th {
    background: #f0f8ef !important;
}

.gradio-container table tr,
.gradio-container table tbody,
.gradio-container table tbody tr,
.gradio-container table tbody td,
.gradio-container .dataframe,
.gradio-container .dataframe *,
.gradio-container .table-wrap,
.gradio-container .table-wrap *,
.gradio-container .cell-wrap,
.gradio-container .cell-wrap *,
.gradio-container [data-testid="dataframe"],
.gradio-container [data-testid="dataframe"] *,
.gradio-container .wrap[data-testid],
.gradio-container .wrap[data-testid] * {
    background-color: #ffffff !important;
    color: var(--agri-ink) !important;
}

.gradio-container .dataframe thead *,
.gradio-container [data-testid="dataframe"] thead *,
.gradio-container .table-wrap thead * {
    background-color: #f0f8ef !important;
    color: var(--agri-ink) !important;
}

.gradio-container h3 {
    color: var(--agri-leaf);
}

.gradio-container label,
.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container option,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .gr-form,
.gradio-container .gradio-container,
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .output-class,
.gradio-container .input-class {
    color: var(--agri-ink) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .block {
    background: #ffffff !important;
}

.gradio-container pre,
.gradio-container code,
.gradio-container .prose pre,
.gradio-container .prose code,
.gradio-container .markdown pre,
.gradio-container .markdown code {
    background: #f5fbf4 !important;
    color: var(--agri-ink) !important;
    border: 1px solid rgba(36, 91, 59, 0.14) !important;
}

.gradio-container .prose table,
.gradio-container .markdown table,
.gradio-container .prose table *,
.gradio-container .markdown table * {
    background-color: #ffffff !important;
    color: var(--agri-ink) !important;
}

.gradio-container .markdown,
.gradio-container .markdown *,
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container p,
.gradio-container span,
.gradio-container li {
    color: var(--agri-ink) !important;
}

.gradio-container .secondary,
.gradio-container .meta-text,
.gradio-container .svelte-1gfkn6j {
    color: var(--agri-muted) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #6b7a70 !important;
}

.diagnosis-card {
    background: linear-gradient(180deg, #ffffff, #f7fbf6) !important;
}

@media (max-width: 900px) {
    .hero-banner {
        padding: 24px 22px;
    }

    .hero-title {
        font-size: 30px;
    }
}
"""


def filter_supported_kwargs(callable_obj, **kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def get_block_theme():
    if gr is None or not hasattr(gr, "themes"):
        return None
    return gr.themes.Soft()


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def dataset_split_counts() -> dict[str, tuple[int, int]]:
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = (
            count_files(DATASET_ROOT / "images" / split),
            count_files(DATASET_ROOT / "labels" / split),
        )
    return counts


def dataset_ready() -> bool:
    return DATASET_YAML.exists() and all((DATASET_ROOT / "images" / split).exists() for split in ("train", "val"))


def default_model_path() -> Path:
    env_path = os.getenv("MBYOLO_APP_MODEL", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_MODEL_PATH


def load_class_names() -> list[str]:
    classes = load_default_class_names(DATASET_ROOT / "classes.txt")
    if classes:
        return classes

    if DATASET_YAML.exists():
        names: list[str] = []
        for line in DATASET_YAML.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            if key.isdigit():
                names.append(value.strip().strip("'\""))
        if names:
            return names

    return []


def extract_model_names(model) -> list[str]:
    names = getattr(model, "names", [])
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    return list(names)


def get_knowledge_base(runtime_class_names: list[str] | None = None) -> KnowledgeBase:
    class_names = tuple(runtime_class_names or APP_STATE["runtime_class_names"])
    if class_names not in KB_CACHE:
        KB_CACHE[class_names] = KnowledgeBase(list(class_names))
    return KB_CACHE[class_names]


def class_count() -> int:
    return len(APP_STATE["runtime_class_names"])


def model_status() -> str:
    if YOLO is None:
        return f"依赖未就绪: {YOLO_IMPORT_ERROR}"
    if default_model_path().exists():
        return "模型已就绪"
    return "未找到 best.pt"


def load_model(weight_path: str):
    if YOLO is None:
        raise RuntimeError(f"ultralytics import failed: {YOLO_IMPORT_ERROR}")
    normalized = str(Path(weight_path).expanduser())
    if normalized not in MODEL_CACHE:
        MODEL_CACHE[normalized] = YOLO(normalized)
    return MODEL_CACHE[normalized]


def initialize_app_state() -> dict:
    runtime_class_names = load_class_names()
    kb = KnowledgeBase(runtime_class_names)
    return {
        "runtime_class_names": runtime_class_names,
        "knowledge_base": kb,
    }


APP_STATE = initialize_app_state()


def configure_local_proxy_bypass() -> None:
    """Prevent proxies from intercepting Gradio's localhost health checks."""
    local_hosts = ["127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        for host in local_hosts:
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


def get_runtime_category_count(category_type: str, kb: KnowledgeBase | None = None) -> int:
    active_kb = kb or get_knowledge_base()
    return sum(
        1
        for class_name in active_kb.runtime_class_names
        if (active_kb.lookup(class_name=class_name) or {}).get("category_type", PEST) == category_type
    )


def extract_predictions(result, model, kb: KnowledgeBase) -> list[dict]:
    predictions: list[dict] = []

    if result.boxes is None:
        return predictions

    runtime_names = extract_model_names(model)
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if 0 <= class_id < len(runtime_names):
            class_name = runtime_names[class_id]
        else:
            class_name = str(class_id)

        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        kb_entry = kb.lookup(class_name=class_name, class_id=class_id) or {}

        predictions.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2],
                "category_type": kb_entry.get("category_type", PEST),
            }
        )

    return predictions


def summarize_predictions(predictions: list[dict], kb: KnowledgeBase) -> list[dict]:
    grouped: dict[str, dict] = {}

    for prediction in predictions:
        class_name = prediction["class_name"]
        current = grouped.setdefault(
            class_name,
            {
                "class_id": prediction["class_id"],
                "class_name": class_name,
                "category_type": prediction["category_type"],
                "count": 0,
                "max_confidence": 0.0,
            },
        )
        current["count"] += 1
        current["max_confidence"] = max(current["max_confidence"], prediction["confidence"])

    summaries: list[dict] = []
    for class_name in sorted(grouped):
        item = grouped[class_name]
        kb_entry = kb.lookup(class_name=class_name, class_id=item["class_id"]) or {}
        level = calculate_risk_level(item["count"], item["max_confidence"], kb_entry.get("risk_rule"))
        item["risk_level"] = level
        item["diagnosis_tag"] = risk_label(level)
        summaries.append(item)

    return summaries


def format_class_stats(class_stats: list[dict]) -> str:
    if not class_stats:
        return "当前模式下未检测到相关目标。"

    lines = [
        "| 类别 | 数量 | 最大置信度 | 诊断标签 |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in class_stats:
        lines.append(
            f"| {item['class_name']} | {item['count']} | {item['max_confidence']:.2f} | {item['diagnosis_tag']} |"
        )
    return "\n".join(lines)


def format_kb_hits(kb_hits: list[dict]) -> str:
    if not kb_hits:
        return "当前结果未命中知识库条目。"

    blocks = []
    for entry in kb_hits:
        blocks.append(
            "\n".join(
                [
                    f"### {entry['class_name']}",
                    f"- 类别类型: {entry['category_type']}",
                    f"- 关联作物: {entry['crop']}",
                    f"- 害虫类型: {entry.get('pest_group') or '未细分'}",
                    f"- 危害/症状: {entry['harm_or_symptom']}",
                    f"- 发生条件: {entry['trigger_conditions']}",
                    f"- 建议措施: {entry['suggested_actions']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def save_runtime_record(record: dict) -> None:
    RUNTIME_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_RECORD_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_unified_inference(input_image, task_type: str, confidence_threshold: float, environment_note: str):
    if not input_image:
        return "请先上传图片。", None, "", "", "", "", ""

    if YOLO is None:
        return f"模型依赖未就绪: {YOLO_IMPORT_ERROR}", None, "", "", "", "", ""

    weight_path = default_model_path()
    if not weight_path.exists():
        return f"未找到模型权重: {weight_path}", None, "", "", "", "", ""

    try:
        model = load_model(str(weight_path))
        runtime_names = extract_model_names(model) or APP_STATE["runtime_class_names"]
        kb = get_knowledge_base(runtime_names)
        results = model.predict(source=input_image, conf=confidence_threshold, save=False, verbose=False)
        result = results[0]
        annotated_image = result.plot()
        if annotated_image is not None and getattr(annotated_image, "ndim", 0) == 3:
            annotated_image = annotated_image[:, :, ::-1]

        raw_predictions = extract_predictions(result, model, kb)
        expected_category = TASK_CATEGORY_MAP.get(task_type, PEST)
        filtered_predictions = [
            prediction for prediction in raw_predictions if prediction["category_type"] == expected_category
        ]
        class_stats = summarize_predictions(filtered_predictions, kb)
        kb_hits = kb.lookup_many(item["class_name"] for item in class_stats)

        structured_context = build_structured_context(
            task_type,
            class_stats,
            kb_hits,
            environment_note=environment_note,
        )
        knowledge_payload = render_local_template(structured_context)
        knowledge_diagnosis_text = format_diagnosis_text(knowledge_payload)

        try:
            ai_payload = call_llm_api(structured_context)
            ai_source = "AI 大模型"
            ai_diagnosis_text = format_diagnosis_text(ai_payload)
        except Exception as exc:
            ai_source = "AI 未启用或调用失败"
            ai_diagnosis_text = (
                f"AI 辅助诊断暂不可用：{exc}\n\n"
                "系统已保留农业知识库诊断结果，可继续用于本地演示和基础防治建议。"
            )

        summary_lines = [
            f"任务类型: {task_type}",
            f"模型权重: {weight_path.name}",
            f"原始检测目标数: {len(raw_predictions)}",
            f"当前模式目标数: {len(filtered_predictions)}",
            f"知识库命中数: {len(kb_hits)}",
            f"AI 诊断状态: {ai_source}",
            "知识库诊断状态: 已生成",
        ]

        supported_count = get_runtime_category_count(expected_category, kb)
        if supported_count == 0:
            summary_lines.append("提示: 当前 Mamba-YOLO 权重没有该模式对应的可识别类别。")
        elif not filtered_predictions:
            summary_lines.append("提示: 当前图片中没有筛出该模式对应的目标，可更换图片或降低阈值复查。")

        save_runtime_record(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task_type": task_type,
                "confidence_threshold": confidence_threshold,
                "raw_predictions": raw_predictions,
                "filtered_predictions": filtered_predictions,
                "class_stats": class_stats,
                "kb_hits": kb_hits,
                "ai_diagnosis_source": ai_source,
                "ai_diagnosis_text": ai_diagnosis_text,
                "knowledge_diagnosis_text": knowledge_diagnosis_text,
                "environment_note": environment_note.strip(),
            }
        )

        return (
            "\n".join(summary_lines),
            annotated_image,
            format_class_stats(class_stats),
            format_kb_hits(kb_hits),
            ai_source,
            ai_diagnosis_text,
            knowledge_diagnosis_text,
        )
    except Exception as exc:
        return f"推理失败: {exc}", None, "", "", "", "", ""


def category_to_value(label: str) -> str:
    return CATEGORY_LABEL_MAP.get(label, "全部")


def format_database_rows(entries: list[dict]) -> list[list[str]]:
    rows = []
    for entry in entries:
        rows.append(
            [
                entry.get("class_name", ""),
                "害虫" if entry.get("category_type", PEST) == PEST else "病害",
                entry.get("crop", ""),
                entry.get("pest_group", ""),
                entry.get("harm_or_symptom", ""),
                entry.get("suggested_actions", ""),
            ]
        )
    return rows


def format_database_markdown(entries: list[dict], keyword: str, category_label: str) -> str:
    if not entries:
        return f"未找到匹配条目。关键词: `{keyword or '空'}`，类别: `{category_label}`。"

    blocks = [
        f"共找到 {len(entries)} 条匹配结果，下面展示前 {min(len(entries), 6)} 条详细说明。",
    ]
    for entry in entries[:6]:
        blocks.append(
            "\n".join(
                [
                    f"### {entry.get('class_name', '')}",
                    f"- 类型: {'害虫' if entry.get('category_type', PEST) == PEST else '病害'}",
                    f"- 作物: {entry.get('crop', '')}",
                    f"- 害虫类型: {entry.get('pest_group') or '未细分'}",
                    f"- 危害/症状: {entry.get('harm_or_symptom', '')}",
                    f"- 发生条件: {entry.get('trigger_conditions', '')}",
                    f"- 建议措施: {entry.get('suggested_actions', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def search_knowledge_database(keyword: str, category_label: str):
    kb = get_knowledge_base()
    entries = kb.search(keyword=keyword or "", category_type=category_to_value(category_label), limit=30)
    return format_database_markdown(entries, keyword or "", category_label), format_database_rows(entries)


def build_database_overview_html() -> str:
    kb = get_knowledge_base()
    counts = kb.category_counts()
    return "\n".join(
        [
            '<section class="quick-card-grid">',
            f'<div class="quick-card"><div class="quick-card-title">知识库总条目</div><div class="quick-card-text">{len(kb.entries)} 条农业病虫害知识条目，包含独立 IP102 害虫知识库与病害种子条目。</div></div>',
            f'<div class="quick-card"><div class="quick-card-title">害虫条目</div><div class="quick-card-text">{counts.get(PEST, 0)} 条。data/ip102_pest_knowledge.json 覆盖 IP102 102 类害虫基础知识。</div></div>',
            f'<div class="quick-card"><div class="quick-card-title">病害条目</div><div class="quick-card-text">{counts.get(DISEASE, 0)} 条。可用于知识检索和 AI 诊断上下文扩展。</div></div>',
            "</section>",
        ]
    )


def build_system_info_markdown() -> str:
    counts = dataset_split_counts()
    total_images = sum(images for images, _ in counts.values())
    total_labels = sum(labels for _, labels in counts.values())
    kb = get_knowledge_base()
    kb_counts = kb.category_counts()

    rows = [
        f"# {PROJECT_EN_NAME}",
        f"## {PROJECT_CN_NAME}",
        "",
        "## 系统状态",
        f"- 模型路径: `{default_model_path()}`",
        f"- 当前前端调用模型: {MODEL_SOURCE_LABEL}",
        f"- 当前状态: {model_status()}",
        f"- 数据集配置: `{DATASET_YAML}`",
        f"- 数据集状态: {'已生成' if dataset_ready() else '未生成'}",
        f"- 当前数据集类别数: {class_count()}",
        f"- 图片总数: {total_images}",
        f"- 标签总数: {total_labels}",
        f"- 农业知识库条目总数: {len(kb.entries)}",
        f"- 知识库害虫条目: {kb_counts.get(PEST, 0)}",
        f"- 知识库病害条目: {kb_counts.get(DISEASE, 0)}",
        "",
        "| Split | Images | Labels |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        images, labels = counts[split]
        rows.append(f"| {split} | {images} | {labels} |")

    rows.extend(
        [
            "",
            "## AI 诊断环境变量",
            "- `.env`: 项目根目录可放置一次性配置文件，后续 `python3 app.py` 会自动读取。",
            "- `OPENAI_API_KEY` 或 `AGRI_LLM_API_KEY`: 大模型 API Key，不配置时 AI 辅助诊断会提示未启用。",
            "- `AGRI_LLM_API_URL`: OpenAI 兼容接口地址，默认 `https://api.openai.com/v1/chat/completions`。",
            "- `AGRI_LLM_MODEL`: 诊断模型名称，默认 `gpt-4o-mini`。",
            "- `AGRI_LLM_TIMEOUT`: 请求超时秒数，默认 `20`。",
            "- `MBYOLO_APP_MODEL`: 可选，自定义前端推理权重路径。",
            "",
            "## 使用说明",
            "1. 上传一张田间图片，选择 `害虫检测` 或 `病害检测`。",
            "2. 当前 IP102 权重主要支持害虫类别；病害检测页保留为知识库与未来统一模型扩展入口。",
            "3. 系统会先做模型推理，再按任务类型过滤结果并命中农业知识库。",
            "4. 若配置了 AI API，会调用模型生成 AI 辅助诊断；农业知识库诊断始终显示。",
            "5. 运行诊断记录会写入 `runtime_records/diagnosis_history.jsonl`。",
        ]
    )
    return "\n".join(rows)


def build_hero_html() -> str:
    counts = dataset_split_counts()
    total_images = sum(images for images, _ in counts.values())
    total_labels = sum(labels for _, labels in counts.values())
    kb = get_knowledge_base()
    kb_counts = kb.category_counts()
    return "\n".join(
        [
            '<section class="hero-banner">',
            '<div class="hero-kicker">Mamba Vision Terminal · IP102 Pest Detection · Knowledge Diagnosis</div>',
            f'<h1 class="hero-title">{PROJECT_EN_NAME}</h1>',
            f'<div class="hero-subtitle">{PROJECT_CN_NAME}基于 Mamba-YOLO-T 与 IP102 数据集构建，集成目标检测、农业知识库检索、AI 辅助诊断与本地知识库诊断，形成适合毕设演示的完整病虫害视觉诊断链路。</div>',
            '<div class="hero-chip-row">',
            f'<div class="hero-chip"><div class="hero-chip-label">当前模型状态</div><div class="hero-chip-value">{model_status()}</div></div>',
            f'<div class="hero-chip"><div class="hero-chip-label">前端当前调用</div><div class="hero-chip-value">{MODEL_SOURCE_LABEL}</div></div>',
            f'<div class="hero-chip"><div class="hero-chip-label">IP102 类别数</div><div class="hero-chip-value">{class_count()}</div></div>',
            f'<div class="hero-chip"><div class="hero-chip-label">图片 / 标签数量</div><div class="hero-chip-value">{total_images} / {total_labels}</div></div>',
            f'<div class="hero-chip"><div class="hero-chip-label">知识库害虫 / 病害</div><div class="hero-chip-value">{kb_counts.get(PEST, 0)} / {kb_counts.get(DISEASE, 0)}</div></div>',
            "</div>",
            "</section>",
        ]
    )


def build_overview_cards_html() -> str:
    return "\n".join(
        [
            '<section class="quick-card-grid">',
            '<div class="quick-card"><div class="quick-card-title">统一识别入口</div><div class="quick-card-text">上传田间图片后统一调用 Mamba-YOLO 权重，并按任务类型过滤害虫或病害结果。</div></div>',
            '<div class="quick-card"><div class="quick-card-title">农业知识库增强</div><div class="quick-card-text">检测结果会命中本地农业数据库，展示关联作物、危害症状、发生条件和防治建议。</div></div>',
            '<div class="quick-card"><div class="quick-card-title">AI + 知识库双路诊断</div><div class="quick-card-text">配置 .env 后调用大模型生成 AI 建议，同时固定显示本地农业知识库诊断，便于人工复核。</div></div>',
            "</section>",
        ]
    )


def create_interface():
    if gr is None:
        detail = f" 真实导入错误: {GRADIO_IMPORT_ERROR}" if GRADIO_IMPORT_ERROR else ""
        raise RuntimeError("未能成功导入 gradio，请先执行 `python3 -m pip install -r requirements-app.txt`。" + detail)

    block_theme = get_block_theme()
    block_kwargs = filter_supported_kwargs(
        gr.Blocks,
        title=f"{PROJECT_EN_NAME} | {PROJECT_CN_NAME}",
        theme=block_theme,
        css=APP_CSS,
    )

    with gr.Blocks(**block_kwargs) as demo:
        gr.HTML(f"<style>{APP_CSS}</style>")
        with gr.Column(elem_id="app-shell"):
            gr.HTML(build_hero_html())
            gr.HTML(build_overview_cards_html())

            with gr.Tab("统一识别入口"):
                gr.HTML('<div class="section-caption">Detection Workspace</div>')
                gr.HTML('<h2 class="section-title">统一识别与智能诊断</h2>')

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes=["panel-card"]):
                        gr.HTML('<div class="panel-heading">输入与任务设置</div>')
                        gr.HTML('<div class="panel-note">上传田间图像，设置任务模式与置信度阈值，然后一键完成识别与诊断。</div>')
                        image_input = gr.Image(
                            label="上传待识别图片",
                            type="filepath",
                            sources=["upload"],
                            height=360,
                        )
                        task_selector = gr.Dropdown(
                            choices=list(TASK_CATEGORY_MAP.keys()),
                            value="害虫检测",
                            label="任务类型",
                        )
                        confidence = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            value=0.25,
                            step=0.05,
                            label="置信度阈值",
                        )
                        environment_note = gr.Textbox(
                            label="环境补充信息",
                            lines=2,
                            placeholder="例如：连阴雨后、棚室湿度较大、叶片背面发现虫体……",
                        )
                        submit_btn = gr.Button("开始识别与诊断", variant="primary", elem_classes=["primary-btn"])

                    with gr.Column(scale=1, elem_classes=["panel-card"]):
                        gr.HTML('<div class="panel-heading">检测结果预览</div>')
                        gr.HTML('<div class="panel-note">系统会绘制目标框，并同步输出当前模式下的结果摘要与诊断来源。</div>')
                        summary_output = gr.Textbox(label="结果摘要", lines=6)
                        diagnosis_source_output = gr.Textbox(label="诊断来源", lines=1)
                        image_output = gr.Image(label="框选结果", interactive=False, height=360)

                with gr.Row(equal_height=True):
                    with gr.Column(elem_classes=["panel-card", "compact-card"]):
                        gr.HTML('<div class="panel-heading">数量统计与风险标签</div>')
                        gr.HTML('<div class="panel-note">按类别输出目标数量、最大置信度和知识库规则计算出的诊断标签。</div>')
                        class_stats_output = gr.Markdown()
                    with gr.Column(elem_classes=["panel-card", "compact-card"]):
                        gr.HTML('<div class="panel-heading">知识库命中</div>')
                        gr.HTML('<div class="panel-note">展示当前识别结果命中的农业知识条目，作为 AI 诊断上下文。</div>')
                        kb_hits_output = gr.Markdown()

                with gr.Row(equal_height=True):
                    with gr.Column(elem_classes=["panel-card", "diagnosis-card"]):
                        gr.HTML('<div class="section-caption">AI Diagnosis</div>')
                        gr.HTML('<div class="panel-heading">AI 辅助诊断</div>')
                        gr.HTML('<div class="panel-note">读取项目根目录 .env 中的 API Key 后调用大模型生成诊断；未配置时显示调用状态。</div>')
                        ai_diagnosis_output = gr.Markdown()

                    with gr.Column(elem_classes=["panel-card", "diagnosis-card"]):
                        gr.HTML('<div class="section-caption">Knowledge Diagnosis</div>')
                        gr.HTML('<div class="panel-heading">农业知识库诊断</div>')
                        gr.HTML('<div class="panel-note">始终基于本地 102 类害虫知识库和风险规则生成，适合作为离线兜底结果。</div>')
                        knowledge_diagnosis_output = gr.Markdown()

                with gr.Column(elem_classes=["panel-card", "diagnosis-card"]):
                    gr.HTML('<div class="section-caption">Diagnosis Layer</div>')
                    gr.HTML('<div class="panel-heading">诊断说明</div>')
                    gr.HTML('<div class="panel-note">左侧为大模型根据检测统计、知识库命中和环境信息生成的综合建议；右侧为本地知识库规则结果，两者同时保留，便于答辩展示和人工复核。</div>')

                click_kwargs = filter_supported_kwargs(
                    submit_btn.click,
                    fn=run_unified_inference,
                    inputs=[image_input, task_selector, confidence, environment_note],
                    outputs=[
                        summary_output,
                        image_output,
                        class_stats_output,
                        kb_hits_output,
                        diagnosis_source_output,
                        ai_diagnosis_output,
                        knowledge_diagnosis_output,
                    ],
                    api_name=False,
                    show_api=False,
                )
                submit_btn.click(**click_kwargs)

            with gr.Tab("农业知识库"):
                gr.HTML('<div class="section-caption">Agricultural Database</div>')
                gr.HTML('<h2 class="section-title">农业病虫害知识库检索</h2>')
                gr.HTML(build_database_overview_html())
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes=["panel-card"]):
                        gr.HTML('<div class="panel-heading">检索条件</div>')
                        gr.HTML('<div class="panel-note">可按类别名称、作物、危害症状或防治建议检索本地农业数据库。</div>')
                        keyword_input = gr.Textbox(label="关键词", placeholder="例如 rice、tomato、螟、晚疫病")
                        category_input = gr.Dropdown(
                            choices=list(CATEGORY_LABEL_MAP.keys()),
                            value="全部",
                            label="类别类型",
                        )
                        search_btn = gr.Button("检索知识库", variant="primary", elem_classes=["primary-btn"])
                    with gr.Column(scale=2, elem_classes=["panel-card"]):
                        gr.HTML('<div class="panel-heading">检索结果</div>')
                        database_markdown = gr.Markdown()
                with gr.Column(elem_classes=["panel-card"]):
                    gr.HTML('<div class="panel-heading">结果表格</div>')
                    database_table = gr.Dataframe(
                        headers=["类别", "类型", "作物", "害虫类型", "危害/症状", "建议措施"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=False,
                    )
                search_kwargs = filter_supported_kwargs(
                    search_btn.click,
                    fn=search_knowledge_database,
                    inputs=[keyword_input, category_input],
                    outputs=[database_markdown, database_table],
                    api_name=False,
                    show_api=False,
                )
                search_btn.click(**search_kwargs)

            with gr.Tab("系统信息"):
                gr.HTML('<div class="section-caption">System Overview</div>')
                gr.HTML('<h2 class="section-title">模型、数据集与诊断链路状态</h2>')
                with gr.Column(elem_classes=["panel-card"]):
                    gr.Markdown(build_system_info_markdown())

    return demo


if __name__ == "__main__":
    configure_local_proxy_bypass()
    demo = create_interface()
    print(f"启动 {PROJECT_EN_NAME} - {PROJECT_CN_NAME} 界面...")
    print("Gradio 会自动选择可用本地端口，终端会显示实际访问地址。")
    launch_kwargs = filter_supported_kwargs(
        demo.launch,
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        show_api=False,
    )
    demo.launch(**launch_kwargs)
