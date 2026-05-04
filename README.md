# [AAAI2025] Mamba YOLO: A Simple Baseline for Object Detection with State Space Model

![Python 3.11](https://img.shields.io/badge/python-3.11-g) ![pytorch 2.3.0](https://img.shields.io/badge/pytorch-2.3.0-blue.svg) [![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)


<div align="center">
  <img src="./asserts/mambayolo.jpg" width="1200px"/>
</div>

## Model Zoo

We've pre-trained YOLO-World-T/M/L from scratch and evaluate on the `MSCOCO2017 val`. 

### Inference on MSCOCO2017 dataset


| model | Params| FLOPs | ${AP}^{val}$ | ${AP}_{{50}}^{val}$ | ${AP}_{{75}}^{val}$ | ${AP}_{{S}}^{val}$ | ${AP}_{{M}}^{val}$ | ${AP}_{{L}}^{val}$ |
| :------------------------------------------------------------------------------------------------------------------- | :------------------- | :----------------- | :--------------: | :------------: | :------------: | :------------: | :-------------: | :------------: |
| [Mamba YOLO-T](./ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml) | 5.8M | 13.2G |       44.5       |          61.2           |          48.2           |          24.7          |          48.8          |          62.0          |
| [Mamba YOLO-M](./ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml) | 19.1M | 45.4G  |       49.1       |          66.5           |          53.5           |          30.6          |          54.0          |          66.4          |
| [Mamba YOLO-L](./ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L.yaml)  | 57.6M | 156.2G |       52.1       |          69.8           |          56.5           |          34.1          |          57.3          |          68.1          |




## Getting started

### 1. Installation

Mamba YOLO is developed based on `torch==2.3.0` `pytorch-cuda==12.1` and `CUDA Version==12.6`. 

#### 2.Clone Project 

```bash
git clone https://github.com/HZAI-ZJNU/Mamba-YOLO.git
```

#### 3.Create and activate a conda environment.
```bash
conda create -n mambayolo -y python=3.11
conda activate mambayolo
```

#### 4. Install torch

```bash
pip3 install torch===2.3.0 torchvision torchaudio
```

#### 5. Install Dependencies
```bash
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

#### 6. Prepare IP102 Dataset

Download the official IP102 archive and put it in the project root as `archive.zip`.
Then convert it into this project's YOLO detection layout:

```bash
python3 scripts/prepare_ip102_dataset.py --zip archive.zip --overwrite
```

The generated dataset structure is:

```text
datasets/pest102/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  pest102.yaml
  classes.txt
```

If the archive contains `classification/train|val|test/<class_id>/*.jpg`, the script creates whole-image pseudo boxes
so the Mamba-YOLO detection pipeline can run. If the archive contains VOC XML annotations, the script converts the real
bounding boxes into YOLO labels automatically.

#### 7. Training Mamba-YOLO-T on IP102

The default training command now uses `datasets/pest102/pest102.yaml`:

```bash
python3 mbyolo_train.py
```

Quick smoke test:

```bash
python3 mbyolo_train.py --epochs 1 --batch_size 2 --workers 0 --device 0
```

Validation and test:

```bash
python3 mbyolo_train.py --task val --device 0
python3 mbyolo_train.py --task test --device 0
```

#### 8. Launch Web Demo

Install the lightweight UI dependency:

```bash
pip install -r requirements-app.txt
```

Start the Gradio app:

```bash
python3 app.py
```

By default the app looks for:

```text
output_dir/pest102/mambayolo_t_pest102/weights/best.pt
```

You can override the inference weight with:

```bash
export MBYOLO_APP_MODEL=/path/to/best.pt
python3 app.py
```

#### Original COCO Training Example
```bash
python mbyolo_train.py --task train --data ultralytics/cfg/datasets/coco.yaml \
 --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
--amp  --project ./output_dir/mscoco --name mambayolo_n
```

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics). The selective-scan from [VMamba](https://github.com/MzeroMiko/VMamba).

## Citations
If you find [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) is useful in your research or applications, please consider giving us a star 🌟 and citing it.

```bibtex
@misc{wang2024mambayolossmsbasedyolo,
      title={Mamba YOLO: SSMs-Based YOLO For Object Detection}, 
      author={Zeyu Wang and Chen Li and Huiying Xu and Xinzhong Zhu},
      year={2024},
      eprint={2406.05835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.05835}, 
}
```
