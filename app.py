from __future__ import annotations

import inspect
import os
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
DATASET_ROOT = PROJECT_ROOT / "datasets" / "pest102"
DATASET_YAML = DATASET_ROOT / "pest102.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output_dir" / "pest102" / "mambayolo_t_pest102" / "weights" / "best.pt"
APP_TITLE = "Mamba-YOLO IP102"
APP_SUBTITLE = "基于状态空间模型的农业害虫检测演示系统"
MODEL_CACHE: dict[str, YOLO] = {}

APP_CSS = """
:root {
    --ink: #1f2937;
    --muted: #667085;
    --line: #e5e7eb;
    --panel: #ffffff;
    --soft: #f7f8fb;
    --blue: #2563eb;
    --green: #198754;
    --coral: #e35d4f;
}

body,
.gradio-container {
    background: #f6f7fb;
    color: var(--ink);
    font-family: "Inter", "SF Pro Display", "PingFang SC", "Microsoft YaHei", sans-serif;
}

.gradio-container {
    max-width: 1240px !important;
}

#app-shell {
    min-height: 94vh;
    padding: 18px 0 20px;
}

.topbar {
    align-items: center;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--line);
    border-radius: 20px;
    display: flex;
    justify-content: space-between;
    margin-bottom: 14px;
    padding: 18px 22px;
}

.brand-title {
    font-size: 28px;
    font-weight: 760;
    letter-spacing: 0;
    line-height: 1.1;
    margin: 0;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-top: 6px;
}

.status-strip {
    display: grid;
    gap: 10px;
    grid-template-columns: repeat(3, minmax(120px, 1fr));
    min-width: 410px;
}

.status-pill {
    background: var(--soft);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 10px 12px;
}

.status-label {
    color: var(--muted);
    font-size: 12px;
}

.status-value {
    font-size: 15px;
    font-weight: 700;
    margin-top: 4px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 12px !important;
}

.section-title {
    font-size: 18px;
    font-weight: 760;
    margin: 2px 0 8px;
}

.section-note {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 10px;
}

.gradio-container .tab-nav button {
    border-radius: 999px !important;
    font-weight: 700 !important;
    margin-right: 8px;
}

.gradio-container .tab-nav button.selected {
    background: #eaf1ff !important;
    color: var(--blue) !important;
}

.gradio-container .gr-button-primary {
    background: var(--blue) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 760 !important;
}

.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-input,
.gradio-container .gr-textbox,
.gradio-container .gr-image,
.gradio-container .gr-dataframe {
    border-radius: 14px !important;
}

@media (max-width: 900px) {
    .topbar {
        align-items: stretch;
        display: block;
    }

    .status-strip {
        grid-template-columns: 1fr;
        margin-top: 14px;
        min-width: 0;
    }
}
"""


def filter_supported_kwargs(callable_obj, **kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def class_count() -> int:
    classes_file = DATASET_ROOT / "classes.txt"
    if classes_file.exists():
        return len([line for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()])
    if DATASET_YAML.exists():
        return sum(1 for line in DATASET_YAML.read_text(encoding="utf-8").splitlines() if line.strip().split(":", 1)[0].isdigit())
    return 0


def default_model_path() -> Path:
    env_path = os.getenv("MBYOLO_APP_MODEL", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_MODEL_PATH


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


def model_ready() -> bool:
    return default_model_path().exists()


def build_header_html() -> str:
    dataset_state = "已就绪" if dataset_ready() else "未生成"
    model_state = "已就绪" if model_ready() else "未找到"
    return "\n".join(
        [
            '<section class="topbar">',
            "<div>",
            f'<h1 class="brand-title">{APP_TITLE}</h1>',
            f'<div class="brand-subtitle">{APP_SUBTITLE}</div>',
            "</div>",
            '<div class="status-strip">',
            f'<div class="status-pill"><div class="status-label">数据集</div><div class="status-value">{dataset_state}</div></div>',
            f'<div class="status-pill"><div class="status-label">类别数</div><div class="status-value">{class_count()}</div></div>',
            f'<div class="status-pill"><div class="status-label">模型权重</div><div class="status-value">{model_state}</div></div>',
            "</div>",
            "</section>",
        ]
    )


def dataset_status_markdown() -> str:
    counts = dataset_split_counts()
    rows = [
        "# 数据集状态",
        "",
        f"- 数据集目录: `{DATASET_ROOT}`",
        f"- 数据配置: `{DATASET_YAML}`",
        f"- 当前状态: {'已生成' if dataset_ready() else '未生成'}",
        f"- 类别数: `{class_count()}`",
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
            "## 一键转换命令",
            "",
            "```bash",
            "python3 scripts/prepare_ip102_dataset.py --zip archive.zip --overwrite",
            "```",
            "",
            "当前 classification 版 IP102 会被转换为整图框伪检测标签；如果压缩包包含 VOC XML 标注，脚本会自动转换真实框。",
        ]
    )
    return "\n".join(rows)


def training_markdown() -> str:
    model_path = default_model_path()
    run_dir = PROJECT_ROOT / "output_dir" / "pest102" / "mambayolo_t_pest102"
    return "\n".join(
        [
            "# 训练与模型",
            "",
            f"- 默认权重路径: `{model_path}`",
            f"- 训练输出目录: `{run_dir}`",
            f"- 模型状态: {'已找到 best.pt' if model_path.exists() else '暂未找到 best.pt'}",
            "",
            "## 默认正式训练",
            "",
            "```bash",
            "python3 mbyolo_train.py",
            "```",
            "",
            "## 快速烟测",
            "",
            "```bash",
            "python3 mbyolo_train.py --epochs 1 --batch_size 2 --workers 0 --device 0",
            "```",
            "",
            "## 验证与测试",
            "",
            "```bash",
            "python3 mbyolo_train.py --task val --device 0",
            "python3 mbyolo_train.py --task test --device 0",
            "```",
            "",
            "可通过环境变量 `MBYOLO_APP_MODEL` 指定前端推理权重。",
        ]
    )


def project_markdown() -> str:
    return "\n".join(
        [
            "# 项目说明",
            "",
            "Mamba-YOLO 将状态空间模型引入 YOLO 检测框架，通过二维选择性扫描增强图像特征建模能力。",
            "",
            "本项目将默认数据集切换为 IP102 害虫数据集，并提供可复现的数据转换脚本、训练入口和演示前端。",
            "",
            "需要注意：当前官网下载的 classification 结构只有图像类别，没有真实目标框。脚本生成的是整图框伪标注，适合先跑通检测流程；如果后续获得 VOC XML 标注，脚本会自动转换真实检测框。",
        ]
    )


def load_model(weight_path: str):
    if YOLO is None:
        raise RuntimeError(f"ultralytics import failed: {YOLO_IMPORT_ERROR}")
    normalized = str(Path(weight_path).expanduser())
    if normalized not in MODEL_CACHE:
        MODEL_CACHE[normalized] = YOLO(normalized)
    return MODEL_CACHE[normalized]


def run_inference(image_path: str | None, weight_path: str, confidence: float):
    if not image_path:
        return "请先上传图片。", None, []

    path = Path(weight_path or default_model_path()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return f"未找到模型权重: {path}", None, []

    try:
        model = load_model(str(path))
        results = model.predict(source=image_path, conf=confidence, save=False, verbose=False)
        result = results[0]
        annotated = result.plot()
        if annotated is not None and getattr(annotated, "ndim", 0) == 3:
            annotated = annotated[:, :, ::-1]

        rows = []
        boxes = result.boxes
        names = result.names or getattr(model, "names", {})
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                confidence_value = float(box.conf[0].item())
                x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                rows.append([class_id, class_name, round(confidence_value, 4), x1, y1, x2, y2])

        summary = f"模型: {path}\n检测目标数: {len(rows)}\n置信度阈值: {confidence:.2f}"
        return summary, annotated, rows
    except Exception as exc:
        return f"推理失败: {exc}", None, []


def create_interface():
    if gr is None:
        detail = f" Import error: {GRADIO_IMPORT_ERROR}" if GRADIO_IMPORT_ERROR else ""
        raise RuntimeError("Gradio is not installed. Run `pip install -r requirements-app.txt`." + detail)

    block_kwargs = filter_supported_kwargs(
        gr.Blocks,
        title=f"{APP_TITLE} | IP102",
        css=APP_CSS,
    )

    with gr.Blocks(**block_kwargs) as demo:
        with gr.Column(elem_id="app-shell"):
            gr.HTML(build_header_html())
            with gr.Tab("图片识别"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, elem_classes=["panel"]):
                        gr.HTML('<div class="section-title">输入设置</div>')
                        gr.HTML('<div class="section-note">上传单张害虫图片，选择训练好的 Mamba-YOLO 权重后开始识别。</div>')
                        image_input = gr.Image(label="上传图片", type="filepath", sources=["upload"], height=300)
                        weight_input = gr.Textbox(label="模型权重路径", value=str(default_model_path()))
                        confidence_input = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="置信度阈值")
                        run_button = gr.Button("开始识别", variant="primary")
                    with gr.Column(scale=1, elem_classes=["panel"]):
                        gr.HTML('<div class="section-title">识别结果</div>')
                        summary_output = gr.Textbox(label="运行摘要", lines=4)
                        image_output = gr.Image(label="框选结果", interactive=False, height=300)
                with gr.Column(elem_classes=["panel"]):
                    gr.HTML('<div class="section-title">预测明细</div>')
                    table_output = gr.Dataframe(
                        headers=["Class ID", "Class Name", "Confidence", "X1", "Y1", "X2", "Y2"],
                        datatype=["number", "str", "number", "number", "number", "number", "number"],
                        interactive=False,
                    )
                click_kwargs = filter_supported_kwargs(
                    run_button.click,
                    fn=run_inference,
                    inputs=[image_input, weight_input, confidence_input],
                    outputs=[summary_output, image_output, table_output],
                    api_name=False,
                    show_api=False,
                )
                run_button.click(**click_kwargs)

            with gr.Tab("数据集状态"):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown(dataset_status_markdown())

            with gr.Tab("训练与模型"):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown(training_markdown())

            with gr.Tab("项目说明"):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown(project_markdown())

    return demo


if __name__ == "__main__":
    demo = create_interface()
    print(f"Starting {APP_TITLE}. Visit http://127.0.0.1:7860")
    launch_kwargs = filter_supported_kwargs(
        demo.launch,
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_api=False,
    )
    demo.launch(**launch_kwargs)
