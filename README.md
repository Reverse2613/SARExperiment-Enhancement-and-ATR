# SAR 图像质量提升与目标识别

基于传统方法（Lee 滤波 + CLAHE）的 SAR 图像质量提升，以及基于深度学习（SAR-YOLO & SAR-ViT）的 SAR 目标自动识别（ATR）实验项目。

> > all wish to know,but few the price will pay.
> >
> > 求知者众，愿付代价者寡。
> >
> > little oratory and much labor.
> >
> > 多实验，少空谈。
> >
> > ---Julius Sumner Miller

## 项目概述

本项目基于 [ATRNet-STAR](https://github.com/waterdisappear/ATRNet-STAR) 大规模 SAR 目标识别数据集，完成两个核心实验：

1. **实验一 —— 图像质量提升**：使用 Lee 滤波抑制相干斑噪声，结合 CLAHE 增强对比度，并通过 ENL、EPI、信息熵等客观指标评估提质效果。
2. **实验二 —— 目标识别**：从零手写白盒 SAR-YOLO（CBAM 注意力机制）和 SAR-ViT（卷积切块嵌入）模型，使用双轨验证策略对比提质前后数据的识别准确率，探究域偏移（Domain Shift）对模型的影响。

## 项目结构

```
SAR/
├── data/
│   ├── ATR_low_quality/          # 原始降质图像（10类，用于实验一输入）
│   ├── ATR_enhanced/             # 提质后图像（实验一输出，实验二验证集）
│   ├── SAR_Cls_Dataset/          # 实验二分类数据集
│   │   ├── train/                # 训练集
│   │   ├── val_original/         # 原始降质图像作为验证集
│   │   └── val_enhanced/         # 提质后图像作为验证集
│   └── weights/                  # 训练保存的最佳模型权重
│       ├── yolo_best.pth
│       └── vit_best.pth
│
├── src/                          # 核心源代码（模块化设计）
│   ├── enhancement/              # 图像提质算法
│   │   ├── filters.py            # Lee 滤波实现
│   │   └── enhancer.py           # CLAHE 对比度增强
│   ├── metrics/                  # 客观评估指标
│   │   └── evaluator.py          # ENL、EPI、信息熵计算
│   ├── recognition/              # 目标识别模型
│   │   └── models/
│   │       ├── sar_yolo.py       # SAR-YOLO（CBAM 注意力机制）
│   │       └── sar_vit.py        # SAR-ViT（卷积切块 Transformer）
│   └── utils/                    # 工具模块
│       ├── data_loader.py        # 图像读写工具
│       └── data_selector.py      # 对实验二的训练数据集清洗与构建
│
├── train/                        # ATRNet-STAR 原始数据集
│
├── main_enhance.py               # 实验一主入口：批量图像提质
├── train.py                      # 实验二主入口：模型训练与双轨验证
├── eval_matrix.py                # 混淆矩阵分析与可视化
├── requirements.txt              # 项目依赖
│
├── vit_log.txt                   # SAR-ViT 训练日志
├── yolo_log.txt                  # SAR-YOLO 训练日志
│
├── Experiment1_Metrics_Report.png                  # 实验一各类别指标对比图
├── Confusion_Matrix_yolo_enhanced.png              # SAR-YOLO 混淆矩阵
├── Confusion_Matrix_vit_enhanced.png               # SAR-ViT 混淆矩阵
```

## 数据集

### 数据来源

本项目基于 **ATRNet-STAR** 数据集，该数据集是一个面向遥感场景下真实目标识别的大规模 SAR 基准数据集，包含 40 个细粒度车辆类别。

### 目标类别（10类）

| 序号 | 类别         | 英文名        |
|------|-------------|---------------|
| 1    | 叉车         | Forklift      |
| 2    | 重型卡车     | Heavy_ST      |
| 3    | 搅拌车       | Mixer_truck   |
| 4    | 皮卡         | Pickup        |
| 5    | 压路机       | Road_roller   |
| 6    | 大客车       | large_bus     |
| 7    | 中型SUV      | medium_SUV    |
| 8    | 中型客车     | medium_bus    |
| 9    | 小型轿车     | mini_car      |
| 10   | 铲车         | shovel_loader |

### 数据划分

- **训练集**：每类最多 `2 × 测试集数量` 张（从 ATRNet-STAR 官方训练集随机抽样，共约 1170 张）
- **验证集（原图）**：原始降质图像，共 585 张
- **验证集（提质图）**：经过 Lee 滤波 + CLAHE 处理后的图像，共 585 张

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 12.1（推荐，用于 GPU 加速训练）
- Windows / Linux（本次实验在Windows下完成，但在Linux下运行也没啥问题）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 实验一：图像质量提升

对 `data/ATR_low_quality/` 下的所有降质图像执行 Lee 滤波 + CLAHE 增强，并输出客观指标对比图。

```bash
python main_enhance.py
```

**处理流程：**

```
原始降质图像 → Lee 滤波（去噪） → CLAHE（对比度增强） → 提质后图像
```

**输出：**
- 提质后的图像保存至 `data/ATR_enhanced/`（保持原目录结构）
- 生成 `Experiment1_Metrics_Report.png`：各类别 ENL、信息熵、EPI 对比柱状图

### 实验二：目标识别模型训练

#### 1. 构建数据集

运行 `data_selector.py` 完成数据清洗、标签归一化和抽样训练数据集：

```bash
python src/utils/data_selector.py # or use "python -m src.utils.data_selector"
```

该代码会：
- 扫描 ATRNet-STAR 官方 XML 标注，进行字符串归一化匹配
- 按 `训练集 ≤ 2 × 测试集` 规则随机抽样
- 构建 `data/SAR_Cls_Dataset/` 标准目录结构

#### 2. 训练模型

```bash
# 训练 SAR-YOLO（基于 CBAM 注意力机制）
python train.py
```

在 [train.py](train.py) 中切换模型：

```python
# 训练 SAR-YOLO
train_model(model_name="yolo", num_epochs=50)

# 训练 SAR-ViT
train_model(model_name="vit", num_epochs=50)
```

**双轨验证策略：** 每个 epoch 同时在原图验证集和提质图验证集上评估，实时输出提质增益/负增益。

#### 3. 混淆矩阵分析

```bash
python eval_matrix.py
```

生成 `Confusion_Matrix_yolo_enhanced.png` 和 `Confusion_Matrix_vit_enhanced.png`。
生成 `Confusion_Matrix_yolo_original.png` 和 `Confusion_Matrix_vit_original.png`。


## 模型架构

### SAR-YOLO（基于 CBAM 注意力机制）

手写搭建的轻量化 CNN 分类网络，包含以下核心组件：

- **ConvBlock**：Conv2d + BatchNorm + SiLU 激活函数
- **CBAM（Convolutional Block Attention Module）**：
  - `ChannelAttention`：通道注意力，决定"看什么"特征（AvgPool + MaxPool → MLP → Sigmoid）
  - `SpatialAttention`：空间注意力，决定"在哪里"看（Mean + Max → 7×7 Conv → Sigmoid），对 SAR 背景杂波抑制至关重要
- **CSPBlock_SAR**：跨阶段局部网络，将输入分为两路（保留 + 深度提取），拼接后经 CBAM 聚焦强散射点
- **Stem**：快速下采样（1→32→64 通道）
- **分类头**：AdaptiveAvgPool2d → Flatten → Dropout(0.3) → Linear

```
输入 [B,1,128,128]
  → Stem (ConvBlock×2, stride=2)
  → Layer1 (ConvBlock + CSPBlock_SAR, 64→128)
  → Layer2 (ConvBlock + CSPBlock_SAR, 128→256)
  → Layer3 (ConvBlock + CSPBlock_SAR, 256→512)
  → AdaptiveAvgPool2d(1) → Flatten → Dropout → Linear(512, 10)
  → 输出 [B, 10]
```

### SAR-ViT（卷积切块 Transformer）

针对 SAR 小样本场景定制的小型 Vision Transformer：

- **ConvPatchEmbedding（CPE）**：用两层卷积（7×7 stride=4 → 3×3 stride=2）替代传统线性切块，保留 SAR 离散亮斑的局部几何拓扑关系
- **Position Embedding**：可学习绝对位置编码 + CLS Token
- **TransformerBlock**：Pre-Norm 多头自注意力 + MLP（GELU 激活），depth=2 防止过拟合
- **分类头**：提取 CLS Token → Linear(embed_dim, 10)

```
输入 [B,1,128,128]
  → ConvPatchEmbedding → [B, 256, 256]
  → Concat CLS Token → [B, 257, 256]
  → + Position Embedding
  → TransformerBlock × 2 (Pre-Norm, 8 heads)
  → LayerNorm → CLS Token → Linear(256, 10)
  → 输出 [B, 10]
```

## 客观评估指标

| 指标 | 英文全称 | 含义 | 评判标准 |
|------|---------|------|---------|
| ENL | Equivalent Number of Looks | 等效视数，衡量背景去噪程度 | 越大越好 |
| EPI | Edge Preservation Index | 边缘保持指数，衡量目标轮廓保留程度 | 越接近 1 越好 |
| SCR | Signal-to-Clutter Ratio | 信杂比 | 越高越好 |
| 信息熵 | Information Entropy | 图像整体信息纯度 | 越低越好 |

## 实验结果

### 图像质量提升

- Lee 滤波有效抑制了相干斑噪声，ENL 显著提升
- CLAHE 在不过度放大噪声的前提下增强了目标对比度
- EPI 维持在约 0.52，实现了噪声抑制与边缘保留的工程折中,其实说白了就是目标的轮廓边缘保持得很差

### 目标识别性能

| 模型 | 验证集最高准确率（原图） | 验证集最高准确率（提质图） | 提质增益 |
|------|----------------------|-------------------------|---------|
| SAR-YOLO | 70.09% | 59.66% | 最高 -20.68% |
| SAR-ViT | 51.11% | 37.95% | 最高 -27.18% |

**关键发现：**

1. **CNN 优于 Transformer**：在小样本 SAR 数据上，SAR-YOLO（70.09%）显著优于 SAR-ViT（51.11%），证明了 CNN 的局部归纳偏置对离散强散射点图像的适应性远超 Transformer。
2. **域偏移（Domain Shift）**：提质后验证集出现了负增益，说明提质算法改变了像素分布的底层概率模型，使训练时学到的特征失效。这指导了后续工作——高质量提质算法应当融入预训练流水线，而非仅作后处理。

## 技术栈

- **图像处理**：OpenCV、SciPy、NumPy
- **深度学习**：PyTorch、torchvision
- **可视化**：Matplotlib、Seaborn
- **评估**：scikit-learn（混淆矩阵）

## 杂记
因为之前都是使用得Linux训练模型，所以很多命令不通用在Windows下，本次实验都是在Windows下完成的。

**类似Linux下的重定向到输出文件**

1.python -u train.py *> train.log  （把输出重定向到train.log文件，但是关闭powershell进程会直接中止train.py的运行,-u参数是直接输出到文件，而不是保存到内存中） 

2.
```

# ========== 第1步：后台启动 ==========
cmd /c "start /b python -u train.py > train.log 2>&1"

# ========== 第2步：记录 PID ==========
wmic process where "name='python.exe'" get ProcessId,CommandLine
# 找到 train.py 对应的 PID，记下来，比如 12345

# ========== 第3步：实时查看日志 ==========
Get-Content train.log -Wait
# 按 Ctrl+C 退出查看，不影响训练进程

# ========== 第4步：训练结束后关闭进程 ==========
Stop-Process -Id 12345 -Force

```

## 参考文献

[1] Liu Y, Li W, Liu L, et al. ATRNet-STAR: A large dataset and benchmark towards remote sensing object recognition in the wild[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[2] O. Kechagias-Stamatis and N. Aouf, "Automatic Target Recognition on Synthetic Aperture Radar Imagery: A Survey," in IEEE Aerospace and Electronic Systems Magazine, vol. 36, no. 3, pp. 56-81, 1 March 2021, doi: 10.1109/MAES.2021.3049857.

[3] Li W, Yang W, Hou Y, et al. SARATR-X: Toward building a foundation model for SAR target recognition[J]. IEEE Transactions on Image Processing, 2025, 34: 869-884.

[4] Li W, Yang W, Liu T, et al. Predicting gradient is better: Exploring self-supervised learning for SAR ATR with a joint-embedding predictive architecture[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2024, 218: 326-338.



## 许可证

本项目仅用于本次课程实验，用于学习与教育参考。