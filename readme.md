# OracleFusion

OracleFusion 是一个面向甲骨文解析与生成的两阶段流程：

- **Stage 1**：基于 Qwen-VL 推理甲骨文字形的部件数量、部件描述及其标注框。
- **Stage 2**：以标注真值（GT）或 Stage 1 的推理结果为条件，生成对应字形。

## 环境配置

建议在项目根目录执行以下命令，并使用 Python 3.9。

### 1. 安装 ms-swift

`ms-swift` 用于加载并推理 Qwen-VL 模型。

```bash
conda create -n swift python=3.9
conda activate swift

git clone https://github.com/modelscope/swift.git
cd swift
git checkout v2.5.0
pip install -e '.[llm]'
```

### 2. 安装 OracleFusion 运行环境

```bash
conda create -n oraclefusion python=3.9
conda activate oraclefusion

conda install -y numpy scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg

pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm \
  easydict visdom freetype-py shapely wandb scipy ftfy accelerate
pip install opencv-python==4.5.4.60 kornia==0.6.8 \
  diffusers==0.16.0 transformers==4.34.0

git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

> 提示：Stage 1 依赖 `ms-swift` 环境；Stage 2 依赖上述 OracleFusion 运行环境。请根据执行阶段切换对应环境。

## 演示数据

为避免上传完整数据集，仓库仅保留 `demo/` 中的单字演示样本。该样本为“岛”，其语义为鸟栖息于山上；目录结构与原始 `data/` 目录保持一致：

```text
demo/
└── data/
    ├── test_box/岛/       # 2 张测试图像
    ├── train_bbox/
    │   ├── 岛.json        # 部首标注框
    │   └── 岛.png          # 对应图像
    └── train_box/岛/      # 训练图像及其增强样本
```

完整数据集未随仓库提供。如需运行完整实验，请自行准备数据，并按以下目录结构放置：

| 目录 | 内容 |
| --- | --- |
| `data/train_box/`、`data/test_box/` | 图像—文本数据 |
| `data/train_bbox/`、`data/test_bbox/` | 图像—文本—部首标注框数据 |
| `data/Y+H/` | 待破译甲骨文数据 |

此外，`infer.py` 和 Stage 2 脚本还会读取 `GT/` 下的标注与字形数据。运行前请确认数据路径与脚本中的配置一致。

## Stage 1：甲骨文解析

运行前，请根据本地环境检查并调整脚本中的模型权重路径、数据路径及 GPU 配置；如输出目录不存在，请先创建：

```bash
mkdir -p poyi results poyi_visul
```

### 标准测试集推理

`infer.py` 在 `data/test_bbox/` 测试集上执行推理，结果保存至 `poyi/inference_result.json`。

```bash
python infer.py
```

### OOD 破译

- `infer_poyi_test.py`：在脚本中 `val_list` 指定的人工筛选已识字集合上进行推理，结果保存至 `poyi/inference_result_poyi.json`。
- `infer_poyi_unseen.py`：在 `Y+H/` 中的未识字数据上进行推理，结果保存至 `poyi/inference_poyi_result_6000.json`，并将可视化结果写入 `results/`。

```bash
python infer_poyi_test.py
python infer_poyi_unseen.py
```

## Stage 2：字形生成

Stage 2 支持两种条件输入：标注真值（GT）或 Stage 1 的预测结果。运行前请根据需要修改脚本中的字体、数据路径、输出目录及 GPU 参数。

### 使用 GT 条件生成

```bash
bash run.sh
```

### 使用 Stage 1 推理结果生成

可选地在命令末尾指定 GPU 编号，默认为 `0`。

```bash
python run_poyi.py [gpu-id]
```

例如，使用第 0 张 GPU：

```bash
python run_poyi.py 0
```

生成结果会写入脚本中 `--log_dir` 指定的目录。