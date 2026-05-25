# Mamba-YOLO-IP102 农业害虫智能检测与诊断系统

本项目是一个面向农业场景的病虫害视觉识别与辅助诊断系统。系统以 Mamba-YOLO 为检测模型基础，将 IP102 害虫数据集转换为 YOLO 检测训练格式，完成害虫目标识别、类别统计、农业知识库检索和 AI/本地双路诊断展示。

项目适合用于本科毕设、课程设计或农业视觉检测原型演示。仓库默认不上传原始数据集和完整训练数据，用户 clone 代码后只需下载官方 IP102 压缩包并运行转换脚本，即可复现本项目的数据目录结构和训练入口。

## 项目特点

- 基于 Mamba-YOLO-T 构建农业害虫检测模型。
- 支持 IP102 官方数据压缩包一键转换为本项目需要的 YOLO 格式。
- 当前 classification 版 IP102 可自动生成整图框伪检测标签，先跑通检测训练流程。
- 若数据包中存在 VOC XML 标注，转换脚本会自动生成真实 YOLO 检测框。
- 提供 Gradio Web 前端，支持图片上传、模型推理、结果预览和类别统计。
- 内置独立 IP102 害虫知识库，可展示作物、害虫类型、危害症状、发生条件和防治建议。
- 支持调用 OpenAI 兼容大模型接口生成 AI 辅助诊断，同时固定展示本地农业知识库诊断。
- 默认训练输出目录和前端权重路径统一，便于服务器训练和本地演示协作。

## 目录结构

```text
Mamba-YOLO-IP102/
├── app.py                              # Gradio 前端与推理诊断入口
├── mbyolo_train.py                     # Mamba-YOLO IP102 训练/验证/测试脚本
├── diagnosis_service.py                # AI 诊断与本地模板兜底逻辑
├── knowledge_base.py                   # 农业知识库构建、检索和风险规则
├── data/
│   ├── ip102_pest_knowledge.json       # 覆盖 IP102 102 类害虫的基础知识库
│   └── agri_disease_knowledge.json     # 病害知识库种子条目
├── scripts/
│   └── prepare_ip102_dataset.py        # IP102 数据集转换脚本
├── datasets/                           # 转换后的数据集，本地生成，不上传 GitHub
├── output_dir/                         # 训练输出，可按需选择性上传关键结果
├── runtime_records/                    # 前端诊断记录，本地生成
├── ultralytics/                        # Mamba-YOLO 使用的 Ultralytics 代码
├── selective_scan/                     # Mamba/SSM CUDA 扩展
├── requirements-app.txt                # 前端演示依赖
└── pyproject.toml                      # 训练项目依赖配置
```

## 环境准备

推荐使用 Linux + NVIDIA GPU 训练。前端页面可以在本地启动，但 Mamba-YOLO 推理通常需要服务器或已正确编译 CUDA 扩展的环境。

```bash
conda create -n mambayolo-ip102 python=3.11 -y
conda activate mambayolo-ip102
```

安装 PyTorch 时请根据服务器 CUDA 版本选择合适命令。示例：

```bash
pip install torch torchvision torchaudio
```

安装项目依赖：

```bash
pip install seaborn thop timm einops
cd selective_scan
pip install -v --no-build-isolation .
cd ..
pip install -v -e .
```

如果服务器上 `selective_scan_cuda_core` 导入时报找不到 `libc10.so`，可设置：

```bash
export TORCH_LIB=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
export LD_LIBRARY_PATH=$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
```

## 数据集准备

本项目使用 IP102 农业害虫数据集。请先从官方或 Kaggle 页面下载数据压缩包，并将其放在项目根目录，命名为：

```text
archive.zip
```

然后运行：

```bash
python3 scripts/prepare_ip102_dataset.py --zip archive.zip --overwrite
```

脚本会生成：

```text
datasets/pest102/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── classes.txt
└── pest102.yaml
```

转换脚本支持两类数据结构：

- `classification/train|val|test/<class_id>/*.jpg`：生成整图框伪检测标签，标签格式类似 `0 0.5 0.5 1.0 1.0`。
- `Detection/VOC2007/Annotations/*.xml`：读取真实目标框并转换为 YOLO 标签。

当前若使用 classification 版 IP102，整图框伪标签主要用于先跑通检测训练流程。后续如果获得真实框标注，应优先使用真实 VOC XML 数据进行训练。

## 模型训练

默认训练命令：

```bash
python3 mbyolo_train.py
```

默认配置：

- 模型结构：`ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml`
- 数据配置：`datasets/pest102/pest102.yaml`
- 输入尺寸：`640`
- 训练轮数：`100`
- batch size：`16`
- 输出目录：`output_dir/pest102/mambayolo_t_pest102`

快速烟测：

```bash
python3 mbyolo_train.py --epochs 1 --batch_size 2 --workers 0 --device 0 --no-amp
```

验证集评估：

```bash
python3 mbyolo_train.py --task val --device 0 --no-amp
```

测试集评估：

```bash
python3 mbyolo_train.py --task test --device 0 --no-amp
```

训练完成后，默认前端会读取：

```text
output_dir/pest102/mambayolo_t_pest102/weights/best.pt
```

也可以通过环境变量指定其它权重：

```bash
export MBYOLO_APP_MODEL=/path/to/best.pt
python3 app.py
```

## 启动 Web 前端

安装前端依赖：

```bash
python3 -m pip install -r requirements-app.txt
```

启动：

```bash
python3 app.py
```

Gradio 会自动选择可用本地端口，终端会显示访问地址。

前端包含三个主要页面：

- `统一识别入口`：上传田间图片，选择任务类型，设置置信度阈值，输出检测图、类别统计、知识库命中和诊断结果。
- `农业知识库`：检索本地农业病虫害数据库，查看作物、危害症状、发生条件和防治建议。
- `系统信息`：查看模型路径、数据集状态、知识库条目数、AI 诊断环境变量和使用说明。

## AI 智能诊断

系统会将检测统计、农业知识库命中结果和用户填写的环境补充信息组装为结构化上下文。前端会同时展示两类结果：

- `AI 辅助诊断`：配置 API Key 后调用 OpenAI 兼容大模型生成。
- `农业知识库诊断`：始终基于本地 `data/ip102_pest_knowledge.json` 和风险规则生成，离线也可使用。

本项目不能使用 Codex/ChatGPT 的账号密码直接登录接入，工程代码应使用 API Key。推荐在项目根目录创建 `.env`，配置一次后每次 `python3 app.py` 会自动读取：

```bash
cp .env.example .env
```

然后打开 `.env` 填入：

```text
OPENAI_API_KEY=你的_API_KEY
AGRI_LLM_API_URL=https://api.openai.com/v1/chat/completions
AGRI_LLM_MODEL=gpt-4o-mini
AGRI_LLM_TIMEOUT=20
```

也可以继续使用环境变量方式：

```bash
export OPENAI_API_KEY=你的_API_KEY
python3 app.py
```

如果没有配置 API Key、网络不可用或接口返回异常，AI 辅助诊断区域会显示不可用原因，农业知识库诊断仍会正常生成，保证答辩演示时页面可用。

诊断记录默认写入：

```text
runtime_records/diagnosis_history.jsonl
```

该文件属于本地运行记录，不建议上传 GitHub。

## GitHub 数据与产物规范

建议上传：

- 代码文件
- 训练脚本
- 数据转换脚本
- 前端和知识库逻辑
- 少量最终报告图、关键结果表或最终演示权重

不建议上传：

- `archive.zip`
- `datasets/`
- 大量中间训练输出
- 临时运行记录
- 本地虚拟环境

本仓库 `.gitignore` 已默认忽略数据集、原始压缩包、运行日志和常见大文件。

## 常见问题

### 1. 为什么 classification 版 IP102 也能训练检测模型？

classification 版 IP102 没有真实目标框。转换脚本会为每张图片生成一个整图框伪标签，使检测训练流程可以跑通。这适合毕设系统联调和原型验证，但检测框精度不等同于真实标注训练。

### 2. 为什么推荐 `--no-amp`？

部分新版本 PyTorch 在加载旧格式权重时会受到 `weights_only=True` 默认行为影响。关闭 AMP 可以绕开 YOLOv8n AMP 检查阶段的额外权重加载问题，适合先稳定完成训练和烟测。

### 3. 前端能打开但推理失败怎么办？

先确认：

- `best.pt` 是否存在。
- `selective_scan` CUDA 扩展是否已安装。
- `LD_LIBRARY_PATH` 是否包含 PyTorch 的 `lib` 目录。
- 当前机器是否支持运行 Mamba-YOLO 的 CUDA 扩展。

如果只是本地查看页面，不一定需要本地成功推理；实际推理可以放到服务器执行。

### 4. 农业知识库从哪里来？

IP102 的 102 类害虫会由 `knowledge_base.py` 自动生成默认知识条目，包括作物推断、危害描述、发生条件、防治建议和风险规则。`data/agri_disease_knowledge.json` 提供少量病害种子条目，用于知识库检索和未来扩展。

## 致谢

本项目基于 Mamba-YOLO 与 Ultralytics 检测框架进行二次开发，并结合 IP102 数据集完成农业害虫检测与诊断系统实现。
