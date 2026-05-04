from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_opt():
    parser = argparse.ArgumentParser(description="Train, validate, or test Mamba-YOLO on the IP102 pest dataset.")
    parser.add_argument("--data", type=str, default="datasets/pest102/pest102.yaml", help="dataset yaml path")
    parser.add_argument(
        "--config",
        type=str,
        default="ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml",
        help="model config path",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="image size in pixels")
    parser.add_argument("--task", default="train", choices=("train", "val", "test"), help="train, val, or test")
    parser.add_argument("--device", default="0", help="cuda device, e.g. 0 or 0,1,2,3, or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers per rank")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--optimizer", default="SGD", help="SGD, Adam, AdamW")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="enable AMP training")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP training")
    parser.add_argument("--project", default="output_dir/pest102", help="save to project/name")
    parser.add_argument("--name", default="mambayolo_t_pest102", help="save to project/name")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision validation")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    return parser.parse_args()


def ensure_file(path: Path, message: str) -> None:
    if not path.exists():
        raise SystemExit(f"{message}: {path}")


def main():
    opt = parse_opt()
    data_path = resolve_path(opt.data)
    model_conf = resolve_path(opt.config)
    project_path = resolve_path(opt.project)

    ensure_file(
        data_path,
        "Dataset yaml not found. Run `python3 scripts/prepare_ip102_dataset.py --zip archive.zip --overwrite` first",
    )
    ensure_file(model_conf, "Model config not found")

    from ultralytics import YOLO

    common_args = {
        "data": str(data_path),
        "imgsz": opt.imgsz,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "device": opt.device,
        "project": str(project_path),
        "name": opt.name,
    }

    model = YOLO(str(model_conf))
    if opt.task == "train":
        return model.train(
            **common_args,
            epochs=opt.epochs,
            optimizer=opt.optimizer,
            amp=opt.amp,
        )
    if opt.task == "val":
        return model.val(**common_args, half=opt.half, dnn=opt.dnn)
    return model.val(**common_args, split="test", half=opt.half, dnn=opt.dnn)


if __name__ == "__main__":
    main()
