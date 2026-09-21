# OracleFusion

OracleFusion 是一个面向甲骨文解析与生成的两阶段流程：

- **Stage 1**：基于 Qwen-VL 推理甲骨文字形的部件数量、部件描述及其标注框。
- **Stage 2**：以标注真值（GT）或 Stage 1 的推理结果为条件，生成对应字形。

## 环境配置


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

## 数据
RMOBS下载链接：https://drive.google.com/drive/folders/1sVfz9SKuHPLU_5ys9MFuXgprY5VlmXNJ?usp=drive_link
如需运行完整实验，请准备数据，并按以下目录结构放置：

| 目录 | 内容 |
| --- | --- |
| `data/train_box/`、`data/test_box/` | 图像—文本数据 |
| `data/train_bbox/`、`data/test_bbox/` | 图像—文本—部首标注框数据 |
| `data/Y+H/` | 待破译甲骨文数据 |

## Stage 1：甲骨文解析


### 标准测试集推理

```bash
python infer.py
```

### OOD 破译

- `infer_poyi_test.py`：在脚本中指定的人工筛选已识字集合上进行推理。
- `infer_poyi_unseen.py`：在未识字数据上进行推理。

```bash
python infer_poyi_test.py
python infer_poyi_unseen.py
```

## Stage 2：字形生成

Stage 2 支持两种条件输入：标注真值（GT）或 Stage 1 的预测结果.

### 使用 GT 条件生成

```bash
bash run.sh
```

### 使用 Stage 1 推理结果生成


```bash
python run_poyi.py 
```

生成结果会写入脚本中 `--log_dir` 指定的目录。
