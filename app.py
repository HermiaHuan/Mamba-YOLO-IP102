from __future__ import annotations

import inspect
import os
from collections import defaultdict
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


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_EN_NAME = "Mamba-YOLO IP102"
PROJECT_CN_NAME = "农业害虫智能检测系统"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "pest102"
DATASET_YAML = DATASET_ROOT / "pest102.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output_dir" / "pest102" / "mambayolo_t_pest102" / "weights" / "best.pt"
MODEL_SOURCE_LABEL = "Mamba-YOLO-T 训练权重"
TASK_CHOICES = ["害虫检测"]
MODEL_CACHE: dict[str, object] = {}

APP_CSS = """
:root {
    --agri-ink: #20342a;
    --agri-muted: #5d6e61;
    --agri-leaf: #2f6b47;
    --agri-moss: #8aa34a;
    --agri-earth: #8a5b3d;
    --agri-cream: #f7f4ea;
    --agri-panel: rgba(255, 255, 255, 0.86);
    --agri-line: rgba(47, 107, 71, 0.14);
    --agri-shadow: 0 18px 45px rgba(48, 72, 53, 0.12);
}

body,
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(138, 163, 74, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(47, 107, 71, 0.12), transparent 26%),
        linear-gradient(180deg, #f4efdf 0%, #edf5eb 48%, #f8f6ef 100%);
    color: var(--agri-ink);
    font-family: "Avenir Next", "SF Pro Display", "PingFang SC", "Microsoft YaHei", sans-serif;
}

.gradio-container {
    max-width: 1220px !important;
}

#app-shell {
    padding-top: 18px;
    padding-bottom: 28px;
}

.hero-banner {
    background:
        linear-gradient(135deg, rgba(32, 52, 42, 0.96), rgba(47, 107, 71, 0.92)),
        linear-gradient(180deg, rgba(138, 163, 74, 0.15), rgba(255, 255, 255, 0));
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 28px;
    box-shadow: var(--agri-shadow);
    color: #f7f6f0;
    overflow: hidden;
    padding: 30px 34px;
    position: relative;
}

.hero-banner::after {
    background:
        radial-gradient(circle, rgba(255, 255, 255, 0.22), transparent 58%);
    content: "";
    height: 240px;
    position: absolute;
    right: -80px;
    top: -80px;
    width: 240px;
}

.hero-kicker {
    color: #cddfb7;
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
    color: rgba(247, 246, 240, 0.84);
    font-size: 15px;
    line-height: 1.8;
    margin-top: 16px;
    max-width: 760px;
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
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 18px;
    padding: 14px 16px;
}

.hero-chip-label {
    color: rgba(247, 246, 240, 0.74);
    font-size: 12px;
    margin-bottom: 6px;
}

.hero-chip-value {
    color: #ffffff;
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
    border-radius: 18px !important;
}

.gradio-container .tabs {
    border: none !important;
}

.gradio-container .tab-nav button {
    border-radius: 999px !important;
    color: var(--agri-muted) !important;
    font-weight: 700 !important;
    margin-right: 8px;
}

.gradio-container .tab-nav button.selected {
    background: rgba(47, 107, 71, 0.12) !important;
    color: var(--agri-leaf) !important;
}

.gradio-container table {
    border-collapse: separate;
    border-spacing: 0;
    color: var(--agri-ink) !important;
    font-size: 14px;
    overflow: hidden;
    width: 100%;
}

.gradio-container table thead tr {
    background: rgba(47, 107, 71, 0.08);
}

.gradio-container table th,
.gradio-container table td {
    border-bottom: 1px solid rgba(47, 107, 71, 0.08);
    color: var(--agri-ink) !important;
    padding: 10px 12px !important;
}

.gradio-container h3 {
    color: var(--agri-leaf);
}

.gradio-container label,
.gradio-container textarea,
.gradio-container input,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .gr-form {
    color: var(--agri-ink) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: var(--agri-muted) !important;
}

.diagnosis-card {
    background: linear-gradient(180deg, rgba(247, 244, 234, 0.9), rgba(255, 255, 255, 0.95));
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
    """Keep only keyword arguments supported by the current Gradio version."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
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
    classes_file = DATASET_ROOT / "classes.txt"
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

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


def class_count() -> int:
    return len(load_class_names())


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


def extract_prediction_rows(result, model) -> list[dict]:
    rows: list[dict] = []
    boxes = result.boxes
    names = result.names or getattr(model, "names", {})

    if boxes is None:
        return rows

    for box in boxes:
        class_id = int(box.cls[0].item())
        if isinstance(names, dict):
            class_name = names.get(class_id, str(class_id))
        else:
            class_name = names[class_id] if class_id < len(names) else str(class_id)
        confidence_value = float(box.conf[0].item())
        x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence_value,
                "box": [x1, y1, x2, y2],
            }
        )
    return rows


def format_class_stats(predictions: list[dict]) -> str:
    if not predictions:
        return "当前图片未检测到害虫目标。"

    grouped: dict[str, dict] = defaultdict(lambda: {"count": 0, "max_confidence": 0.0, "class_id": ""})
    for item in predictions:
        current = grouped[item["class_name"]]
        current["count"] += 1
        current["class_id"] = item["class_id"]
        current["max_confidence"] = max(current["max_confidence"], item["confidence"])

    lines = [
        "| 类别 | Class ID | 数量 | 最大置信度 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for class_name in sorted(grouped):
        item = grouped[class_name]
        lines.append(
            f"| {class_name} | {item['class_id']} | {item['count']} | {item['max_confidence']:.2f} |"
        )
    return "\n".join(lines)


def format_dataset_hits(predictions: list[dict]) -> str:
    if not predictions:
        return "当前结果未命中可展示类别。若图片清晰但没有结果，可适当降低置信度阈值后重试。"

    class_names = []
    for item in predictions:
        if item["class_name"] not in class_names:
            class_names.append(item["class_name"])

    blocks = []
    for class_name in class_names[:5]:
        blocks.append(
            "\n".join(
                [
                    f"### {class_name}",
                    "- 数据来源: IP102 农业害虫数据集",
                    "- 模型链路: Mamba-YOLO-T 检测权重",
                    "- 结果解释: 当前类别由模型根据目标区域特征与训练类别分布给出。",
                    "- 使用建议: 结合田间实际虫体形态、作物受害部位和多张图片进行复核。",
                ]
            )
        )
    return "\n\n".join(blocks)


def format_diagnosis(predictions: list[dict], confidence_threshold: float, environment_note: str) -> str:
    if not predictions:
        return (
            "诊断结论：当前图片未检出有效害虫目标。\n\n"
            "风险等级：低风险或待复核\n\n"
            "防治建议：建议重新上传光照充足、目标清晰的近景图片，必要时降低置信度阈值复查。\n\n"
            "注意事项：检测结果仅作为辅助判断，应结合田间实际情况和人工观察。"
        )

    grouped: dict[str, dict] = defaultdict(lambda: {"count": 0, "max_confidence": 0.0})
    for item in predictions:
        current = grouped[item["class_name"]]
        current["count"] += 1
        current["max_confidence"] = max(current["max_confidence"], item["confidence"])

    primary_name, primary_info = max(grouped.items(), key=lambda pair: (pair[1]["count"], pair[1]["max_confidence"]))
    risk_level = "中风险" if primary_info["max_confidence"] >= max(confidence_threshold, 0.5) else "低风险"
    env_text = environment_note.strip() or "未填写环境补充信息"
    return (
        f"诊断结论：模型检测到 {primary_info['count']} 个主要疑似害虫类别 {primary_name}，"
        f"最高置信度为 {primary_info['max_confidence']:.2f}。\n\n"
        f"风险等级：{risk_level}\n\n"
        "防治建议：建议优先核对虫体形态和作物受害部位；若田间连续出现同类目标，"
        "可结合当地植保方案开展监测、诱捕或精准防治。\n\n"
        f"环境补充：{env_text}\n\n"
        "注意事项：当前项目使用 IP102 分类数据生成的整图框伪检测标签完成流程跑通，"
        "检测结果适合毕设演示和方法验证，正式应用仍建议使用真实框标注继续训练。"
    )


def run_unified_inference(
    input_image: str | None,
    task_type: str,
    confidence_threshold: float,
    environment_note: str,
):
    if not input_image:
        return "请先上传图片。", None, "", "", "", ""

    if task_type != "害虫检测":
        return "当前项目仅支持 IP102 害虫检测。", None, "", "", "", ""

    weight_path = default_model_path()
    if not weight_path.exists():
        return f"未找到模型权重: {weight_path}", None, "", "", "", ""

    try:
        model = load_model(str(weight_path))
        results = model.predict(source=input_image, conf=confidence_threshold, save=False, verbose=False)
        result = results[0]
        annotated = result.plot()
        if annotated is not None and getattr(annotated, "ndim", 0) == 3:
            annotated = annotated[:, :, ::-1]

        predictions = extract_prediction_rows(result, model)
        summary_lines = [
            f"任务类型: {task_type}",
            f"模型权重: {weight_path.name}",
            f"检测目标数: {len(predictions)}",
            f"置信度阈值: {confidence_threshold:.2f}",
            f"环境补充: {environment_note.strip() or '未填写'}",
        ]

        return (
            "\n".join(summary_lines),
            annotated,
            format_class_stats(predictions),
            format_dataset_hits(predictions),
            "local_template",
            format_diagnosis(predictions, confidence_threshold, environment_note),
        )
    except Exception as exc:
        return f"推理失败: {exc}", None, "", "", "", ""


def build_system_info_markdown() -> str:
    counts = dataset_split_counts()
    total_images = sum(images for images, _ in counts.values())
    total_labels = sum(labels for _, labels in counts.values())

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
        f"- 类别总数: {class_count()}",
        f"- 图片总数: {total_images}",
        f"- 标签总数: {total_labels}",
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
            "## 使用说明",
            "1. 上传一张害虫图片，任务类型保持 `害虫检测`。",
            "2. 设置置信度阈值，阈值越低越容易检出，误检也可能增多。",
            "3. 可填写田间环境补充信息，系统会把它写入本地模板诊断。",
            "4. 前端默认调用 `output_dir/pest102/mambayolo_t_pest102/weights/best.pt`。",
            "5. 可通过环境变量 `MBYOLO_APP_MODEL` 指定其它权重路径。",
            "",
            "## 常用命令",
            "```bash",
            "python3 scripts/prepare_ip102_dataset.py --zip archive.zip --overwrite",
            "python3 mbyolo_train.py",
            "python3 app.py",
            "```",
        ]
    )
    return "\n".join(rows)


def build_hero_html() -> str:
    counts = dataset_split_counts()
    total_images = sum(images for images, _ in counts.values())
    total_labels = sum(labels for _, labels in counts.values())
    return "\n".join(
        [
            '<section class="hero-banner">',
            '<div class="hero-kicker">Mamba Vision Terminal · IP102 Pest Detection</div>',
            f'<h1 class="hero-title">{PROJECT_EN_NAME}</h1>',
            f'<div class="hero-subtitle">{PROJECT_CN_NAME}面向农业害虫图像识别场景，基于 Mamba-YOLO 完成目标检测、结果统计与本地模板诊断展示。界面风格与 pest-disease-vision 保持一致，适合本地演示与毕设答辩。</div>',
            '<div class="hero-chip-row">',
            f'<div class="hero-chip"><div class="hero-chip-label">当前模型状态</div><div class="hero-chip-value">{model_status()}</div></div>',
            f'<div class="hero-chip"><div class="hero-chip-label">前端当前调用</div><div class="hero-chip-value">{MODEL_SOURCE_LABEL}</div></div>',
            f'<div class="hero-chip"><div class="hero-chip-label">权重类别总数</div><div class="hero-chip-value">{class_count()}</div></div>',
            f'<div class="hero-chip"><div class="hero-chip-label">图片 / 标签数量</div><div class="hero-chip-value">{total_images} / {total_labels}</div></div>',
            "</div>",
            "</section>",
        ]
    )


def build_overview_cards_html() -> str:
    return "\n".join(
        [
            '<section class="quick-card-grid">',
            '<div class="quick-card"><div class="quick-card-title">统一识别入口</div><div class="quick-card-text">同一页面完成图片上传、阈值设置、Mamba-YOLO 推理与结果回传，保持与原病虫害视觉系统一致的交互链路。</div></div>',
            '<div class="quick-card"><div class="quick-card-title">检测结果统计</div><div class="quick-card-text">输出框选图、类别数量、最大置信度和类别说明，方便观察模型对害虫目标的识别情况。</div></div>',
            '<div class="quick-card"><div class="quick-card-title">答辩演示友好</div><div class="quick-card-text">权重不存在或推理失败时页面仍可打开；训练好 best.pt 后即可直接进行本地或服务器演示。</div></div>',
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
                            choices=TASK_CHOICES,
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
                        gr.HTML('<div class="panel-heading">数量统计与类别概率</div>')
                        gr.HTML('<div class="panel-note">按类别输出目标数量和最大置信度，便于观察识别结果。</div>')
                        class_stats_output = gr.Markdown()
                    with gr.Column(elem_classes=["panel-card", "compact-card"]):
                        gr.HTML('<div class="panel-heading">类别与数据说明</div>')
                        gr.HTML('<div class="panel-note">展示当前识别类别对应的数据集来源、模型链路和使用建议。</div>')
                        kb_hits_output = gr.Markdown()

                with gr.Column(elem_classes=["panel-card", "diagnosis-card"]):
                    gr.HTML('<div class="section-caption">Diagnosis Layer</div>')
                    gr.HTML('<div class="panel-heading">本地模板诊断输出</div>')
                    gr.HTML('<div class="panel-note">根据检测统计和环境补充信息生成演示用诊断文本，便于答辩展示完整链路。</div>')
                    diagnosis_output = gr.Markdown()

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
                        diagnosis_output,
                    ],
                    api_name=False,
                    show_api=False,
                )
                submit_btn.click(**click_kwargs)

            with gr.Tab("系统信息"):
                gr.HTML('<div class="section-caption">System Overview</div>')
                gr.HTML('<h2 class="section-title">模型与数据集状态</h2>')
                with gr.Column(elem_classes=["panel-card"]):
                    gr.Markdown(build_system_info_markdown())

    return demo


if __name__ == "__main__":
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
