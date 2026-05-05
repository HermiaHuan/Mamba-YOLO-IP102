from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLASS_FILE = PROJECT_ROOT / "datasets" / "pest102" / "classes.txt"
DISEASE_SEED_FILE = PROJECT_ROOT / "data" / "agri_disease_knowledge.json"

PEST = "pest"
DISEASE = "disease"


def load_default_class_names(class_file: Path | None = None) -> list[str]:
    source = class_file or DEFAULT_CLASS_FILE
    if not source.exists():
        return []
    return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_seed_entries(seed_path: Path) -> list[dict]:
    if not seed_path.exists():
        return []

    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("entries", []))
    if isinstance(payload, list):
        return list(payload)
    return []


def _infer_crop(class_name: str, category_type: str) -> str:
    normalized = class_name.lower()
    keyword_map = [
        ("rice", "水稻"),
        ("corn", "玉米"),
        ("maize", "玉米"),
        ("wheat", "小麦"),
        ("beet", "甜菜"),
        ("alfalfa", "苜蓿"),
        ("grape", "葡萄"),
        ("citr", "柑橘"),
        ("mango", "芒果"),
        ("potato", "马铃薯"),
        ("tomato", "番茄"),
        ("apple", "苹果"),
        ("soybean", "大豆"),
        ("cotton", "棉花"),
        ("cabbage", "甘蓝"),
        ("tea", "茶树"),
    ]
    for keyword, crop in keyword_map:
        if keyword in normalized:
            return crop
    if category_type == DISEASE:
        return "对应病害作物"
    return "农田常见作物"


def _default_risk_rule(category_type: str) -> dict:
    if category_type == DISEASE:
        return {
            "count_thresholds": {"medium": 1, "high": 2, "critical": 4},
            "confidence_thresholds": {"medium": 0.35, "high": 0.55, "critical": 0.75},
        }

    return {
        "count_thresholds": {"medium": 1, "high": 3, "critical": 6},
        "confidence_thresholds": {"medium": 0.30, "high": 0.50, "critical": 0.70},
    }


def _generate_default_entry(class_id: int, class_name: str, category_type: str = PEST) -> dict:
    crop = _infer_crop(class_name, category_type)
    if category_type == DISEASE:
        harm_or_symptom = f"{class_name} 常表现为叶片、茎秆或果实出现病斑、霉层、腐烂或失绿症状。"
        trigger_conditions = "高湿、连作、通风不良或病残体未及时清理时更容易发生和扩展。"
        suggested_actions = "优先清除病残叶片，优化通风与水肥管理，并按作物类型选择对症药剂轮换防治。"
    else:
        harm_or_symptom = f"{class_name} 可能啃食叶片、茎秆、果实或吸食汁液，造成产量损失和品质下降。"
        trigger_conditions = "高温高湿、田间杂草较多、监测频率不足或前期虫源积累时风险会升高。"
        suggested_actions = "建议结合诱捕监测、清园除草、生物防治与对口药剂轮换的综合防控措施。"

    return {
        "class_id": class_id,
        "class_name": class_name,
        "category_type": category_type,
        "crop": crop,
        "harm_or_symptom": harm_or_symptom,
        "trigger_conditions": trigger_conditions,
        "suggested_actions": suggested_actions,
        "risk_rule": _default_risk_rule(category_type),
    }


def _normalize_entry(entry: dict) -> dict:
    class_name = str(entry.get("class_name", "")).strip()
    category_type = entry.get("category_type", PEST)
    class_id = entry.get("class_id", -1)

    fallback = _generate_default_entry(-1, class_name, category_type)
    return {
        "class_id": class_id if isinstance(class_id, int) else -1,
        "class_name": class_name,
        "category_type": category_type,
        "crop": entry.get("crop") or _infer_crop(class_name, category_type),
        "harm_or_symptom": entry.get("harm_or_symptom") or fallback["harm_or_symptom"],
        "trigger_conditions": entry.get("trigger_conditions") or fallback["trigger_conditions"],
        "suggested_actions": entry.get("suggested_actions") or fallback["suggested_actions"],
        "risk_rule": entry.get("risk_rule") or _default_risk_rule(category_type),
    }


class KnowledgeBase:
    def __init__(
        self,
        runtime_class_names: Sequence[str] | None = None,
        *,
        seed_path: Path | None = None,
        persist_path: Path | None = None,
    ) -> None:
        self.runtime_class_names = list(runtime_class_names or load_default_class_names())
        self.seed_entries = _load_seed_entries(seed_path or DISEASE_SEED_FILE)
        self.entries = self._build_entries()
        self.index_by_name = {entry["class_name"]: entry for entry in self.entries}
        if persist_path is not None:
            self.export(persist_path)

    def _build_entries(self) -> list[dict]:
        merged: dict[str, dict] = {}
        for entry in self.seed_entries:
            normalized = _normalize_entry(entry)
            if normalized["class_name"]:
                merged[normalized["class_name"]] = normalized

        runtime_entries: list[dict] = []
        for class_id, class_name in enumerate(self.runtime_class_names):
            if class_name in merged:
                entry = deepcopy(merged[class_name])
                entry["class_id"] = class_id
                entry["category_type"] = entry.get("category_type", PEST)
            else:
                entry = _generate_default_entry(class_id, class_name, PEST)
                merged[class_name] = entry
            runtime_entries.append(entry)

        extra_entries = [entry for name, entry in merged.items() if name not in self.runtime_class_names]
        return runtime_entries + extra_entries

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def lookup(self, class_name: str | None = None, class_id: int | None = None) -> dict | None:
        if class_name and class_name in self.index_by_name:
            return deepcopy(self.index_by_name[class_name])

        if class_id is not None and 0 <= class_id < len(self.runtime_class_names):
            return self.lookup(class_name=self.runtime_class_names[class_id])

        return None

    def lookup_many(self, class_names: Iterable[str]) -> list[dict]:
        hits: list[dict] = []
        seen: set[str] = set()
        for class_name in class_names:
            if class_name in seen:
                continue
            seen.add(class_name)
            hit = self.lookup(class_name=class_name)
            if hit:
                hits.append(hit)
        return hits

    def validate_coverage(self, class_names: Sequence[str] | None = None) -> list[str]:
        targets = class_names or self.runtime_class_names
        return [class_name for class_name in targets if class_name not in self.index_by_name]

    def category_counts(self) -> dict[str, int]:
        counts = {PEST: 0, DISEASE: 0}
        for entry in self.entries:
            category_type = entry.get("category_type", PEST)
            counts[category_type] = counts.get(category_type, 0) + 1
        return counts

    def search(self, keyword: str = "", category_type: str = "全部", limit: int = 20) -> list[dict]:
        normalized_keyword = keyword.strip().lower()
        results = []
        for entry in self.entries:
            if category_type != "全部" and entry.get("category_type", PEST) != category_type:
                continue
            haystack = " ".join(
                str(entry.get(field, ""))
                for field in ("class_name", "crop", "harm_or_symptom", "trigger_conditions", "suggested_actions")
            ).lower()
            if normalized_keyword and normalized_keyword not in haystack:
                continue
            results.append(deepcopy(entry))
            if len(results) >= limit:
                break
        return results
