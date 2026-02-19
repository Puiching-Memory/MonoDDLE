# 基于视觉大模型度量深度蒸馏的单目3D目标检测研究计划书

## 一、 研究背景与动机 (Motivation)

单目3D目标检测（Monocular 3D Object Detection）的核心难点在于从单张RGB图像中恢复丢失的深度信息。现有的主流方法（如 MonoDLE, CenterNet3D）通常依赖稀疏的 LiDAR 点云真值（Ground Truth）进行监督训练。然而，LiDAR 真值存在以下局限性：
1. **稀疏性**：仅在物体表面有深度值，无法监督背景和物体内部的几何形状。
2. **数据获取成本高**：强依赖昂贵的采集设备，难以扩展到大规模无雷达数据。

与此同时，近期出现的视觉基础模型（Vision Foundation Models），特别是 **Depth Anything V3 (DA3)**，展现了惊人的深度估计能力。特别是 `DA3NESTED-GIANT-LARGE-1.1` 模型，能够输出高质量的**绝对度量深度（Metric Depth）**，其稠密性和泛化性远超传统监督方法。

本项目旨在**利用 Depth Anything V3 的绝对深度作为“软标签”或“密集监督信号”，通过知识蒸馏（Knowledge Distillation）的方式，指导轻量级单目检测器（如 MonoDLE）学习更鲁棒的深度特征**，从而在不增加推理成本的前提下显著提升检测精度。

---

## 二、 研究目标 (Objectives)

1. **构建深度蒸馏框架**：无需修改检测器的主干网络（Backbone），仅通过改进训练损失函数（Loss Function），将 DA3 的深度知识转移给检测器。
2. **解决深度尺度模糊**：利用 DA3 的度量深度能力，辅助单目检测器更好地回归绝对距离。
3. **实现低成本提点**：在 KITTI 数据集上，通过引入蒸馏损失，提升 AP3D/APBEV 指标，同时保持推理速度不变（Inference time unchanged）。

---

## 三、 核心方法设计 (Methodology)

本项目将在现有的 `MonoDDLE` 代码库基础上进行改进，核心策略是 **“离线生成 + 在线蒸馏”**。

### 1. 数据预处理：离线深度生成
利用 `Depth Anything V3` (DA3NESTED-GIANT-LARGE-1.1) 预处理 KITTI 训练集：
- 输入：单目 RGB 图像（KITTI 原始图像）。
- 输出：与图像分辨率对齐的**绝对度量深度图**，单位为米。
- 工具：`tools/generate_da3_depth.py`。
- 存储：保存为 `.npz` 格式（包含 `depth` 键值），路径：`data/KITTI/DA3_depth_results`。

### 2. 模型训练：密集深度蒸馏损失 (Dense Depth Distillation Loss)
在 `MonoDLE` 原有的损失函数基础上，新增一个蒸馏损失项 $L_{distill}$。

$$L_{total} = L_{cls} + L_{bbox} + L_{dim} + \lambda \cdot L_{distill}$$

其中 $L_{distill}$ 设计为：
- **前景加权 (Foreground-Aware)**：重点计算物体 Bounding Box 区域内的深度误差，背景区域赋予较低权重，避免背景噪声干扰。
- **损失形式**：采用 $L1$ Loss 或 $Scale-Invariant Logarithmic (SILog)$ Loss 来衡量预测深度图 $D_{pred}$ 与 DA3 深度图 $D_{teacher}$ 的差异。

### 4. 实验分析与可视化系统
为了深入分析蒸馏效果并增强实验结果的可解释性，本项目将构建一套基于 **HTML + ECharts** 的交互式可视化看板：
- **深度一致性分析**：利用 ECharts 的热力图（Heatmap）与 3D 散点图，在大规模测试集上可视化预测深度与 DA3/真值的分布偏差。
- **性能多维对比**：使用雷达图（Radar Chart）对比不同 $\lambda$ 权重下 AP3D、APBEV、Heading 及距离误差等多项指标的平衡。
- **交互式 Case Study**：开发 HTML 页面，支持鼠标悬停查看特定物体的 3D 回归误差与深度响应图，便于错误分析。

---

## 四、 实施计划 (Implementation Roadmap)

Python 环境：`/desay120T/ct/dev/uid01954/MonoDDLE/.venv`

### 第一阶段：环境搭建与数据准备 (已完成)
- [x] 跑通 `MonoDDLE` 基线训练代码 (KITTI数据集)。
- [x] 编写并运行 `tools/generate_da3_depth.py`，调用 `Depth-Anything-3` 模型。
- [x] 对 KITTI 训练集生成深度图伪标签 (`.npz` 格式)，完成可视化验证。

### 第二阶段：损失函数开发 (进行中)
- [ ] 修改数据加载器 (`lib/datasets/kitti/kitti_dataset.py`)，支持同步读取 `.npz` 深度文件。
- [ ] 修改模型 (`lib/models/centernet3d.py`)，确保输出 dense depth map（现有 CenterNet 结构已支持，需确认分辨率对齐问题）。
- [ ] 修改损失函数 (`lib/losses/loss_function.py`)，实现 $L_{distill}$

### 第三阶段：实验与调优 (第4-5周)
- [ ] **实验 A (Baseline)**：复现原始 MonoDLE 精度。
- [ ] **实验 B (Distill)**：加入 $L_{distill}$ 进行训练，调整权重 $\lambda$ (如 0.1, 1.0)。
- [ ] **消融分析**：对比全图蒸馏 vs. 仅前景蒸馏的效果差异。

### 第四阶段：数据分析与论文撰写 (第6周)
- [ ] 依托 **HTML + ECharts** 技术栈构建实验分析看板，绘制多维度指标对比图与深度分布直方图。
- [ ] 整理实验数据，输出交互式可视化结果（对比预测深度、真值与 DA3 教师深度）。
- [ ] 撰写毕业论文。

---

## 五、 预期成果与创新点

1. **创新点**：提出了一种基于**视觉基础模型（VFM）绝对深度蒸馏**的单目3D检测训练范式，验证了 VFM 在几何感知任务中的迁移能力。
2. **工程价值**：提供了一种“即插即用”的性能提升方案，无需修改部署模型结构，极易落地。
3. **预期指标**：在 KITTI Easy/Moderate 难度下的 AP3D 指标预期提升 1-3 个百分点。
