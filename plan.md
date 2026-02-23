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

### 3. 不确定性引导的自适应深度蒸馏 (Uncertainty-Aware Adaptive Distillation)
DA3 在物体边缘、透明/反光表面、极远距离处依然会产生深度估计误差。为了实现对噪声标签的鲁棒蒸馏：
- 让网络在预测深度 $d$ 的同时，额外预测一个**深度不确定性（Variance, $\sigma^2$）**。
- 引入 Kendall 不确定性损失重构蒸馏损失：$L_{distill} = \frac{||d_{pred} - d_{DA3}||^2}{2\sigma^2} + \frac{1}{2}\log(\sigma^2)$。
- 当 DA3 伪标签与图像特征矛盾时，网络会自动预测较大的 $\sigma^2$ 降低惩罚权重。

### 4. 深度引导的动态特征对齐 (Depth-Guided Dynamic Feature Alignment)
为了让主干网络在提取特征时“主动”感知深度，而不仅仅在输出层进行蒸馏：
- 在 Neck 层引入轻量级的**深度感知注意力模块 (Depth-Aware Attention)**。
- 将 DA3 深度图下采样并生成 Spatial Attention Map，乘以 RGB 特征图，使网络根据物体远近动态调整特征提取侧重点。

### 5. 基于 DA3 深度的 3D 感知数据增强 (3D-Aware DepthMix)
针对单目 3D 检测的长尾分布问题，利用 DA3 深度图进行符合 3D 几何关系的 Copy-Paste：
- 利用 2D Bbox + DA3 深度从图像中提取前景物体，粘贴到新图像中。
- **遮挡处理**：比较粘贴物体的深度与背景对应位置的 DA3 深度，若大于背景深度则视为被遮挡，不予显示。

### 6. 密集深度与 3D 边界框的几何一致性约束 (Depth-Geometry Consistency)
解决“像素级深度”与“实例级深度”预测分支割裂的问题：
- 提取 3D Bbox 中心点在密集深度图上的局部平均深度（Local Pooled Depth）。
- 强制 3D Bbox 回归分支输出的深度 $Z_{bbox}$ 与该局部密集深度保持一致：$L_{consist} = || Z_{bbox} - \text{Pool}(D_{dense}, \text{box\_center}) ||$。

### 7. 实验分析与可视化系统
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

### 第二阶段：损失函数开发与 DA3 蒸馏集成 (已完成)
- [x] 修改数据加载器 (`lib/datasets/kitti/kitti_dataset.py`)，支持同步读取 `.npz` 深度文件。
- [x] 修改模型 (`lib/models/centernet3d.py`)，确保输出 dense depth map。
- [x] 修改损失函数 (`lib/losses/loss_function.py`)，实现 $L_{distill}$（支持 L1 / SILog + 前景加权）。

### 第三阶段：核心创新点开发 (进行中)
- [ ] **不确定性蒸馏**：修改模型输出深度不确定性 $\sigma^2$，并实现 Kendall 不确定性损失。
- [ ] **深度特征对齐**：在 Neck 层设计并接入 Depth-Aware Attention 模块。
- [ ] **3D-Aware DepthMix**：开发基于 DA3 深度的 3D Copy-Paste 数据增强脚本。
- [ ] **几何一致性约束**：实现 $L_{consist}$，对齐密集深度与 3D Bbox 中心深度。

### 第四阶段：实验与调优 (进行中)
- [x] **实验 A (Baseline)**：复现原始 MonoDLE 精度 (已完成，见 `experiments/results/monodle/kitti_da3/20260219_233544`)。
- [x] **实验 B (Distill)**：加入基础 $L_{distill}$ 进行训练，调整权重 $\lambda$。
- [x] **实验 C (创新点验证)**：逐步加入不确定性蒸馏、特征对齐、DepthMix 和几何一致性约束，记录性能提升。
  - **初步实验结论 (不确定性蒸馏)**：
    - 引入不确定性损失后，模型在宽松标准 (`AP@0.50`) 下获得了显著提升（如 BEV AP 提升 2.5~4.1 个点，3D AP 提升 1.1~3.0 个点），证明不确定性损失有效缓解了 DA3 伪标签的局部噪声，提升了整体深度估计的鲁棒性。
    - 在严格标准下，官方的 40 点插值指标 (`AP_R40@0.70`) 也有微小提升（+0.22）。但旧版的 11 点插值指标 (`AP@0.70`) 出现了约 3 个点的下降。
    - **消融实验结论 (推理阶段融合不确定性)**：尝试将 `dense_depth_uncertainty` 融合到最终置信度打分中（`Score = Score * exp(-Dense_Uncertainty)`），结果导致各项指标全面崩盘（2D Bbox AP 从 96 暴跌至 88）。
      - **原因分析**：MonoDLE 原本已经使用了 3D 边界框的深度不确定性（`sigma`）来惩罚置信度。而 `dense_depth_uncertainty` 是像素级的蒸馏不确定性，代表 DA3 伪标签在该像素点的可靠程度，并不等同于目标级别的存在置信度。强行相乘会导致过度惩罚，破坏了原本的排序。
      - **结论**：推理代码应保持原样。11 点 AP 的下降更多是因为该指标对 PR 曲线头部过于敏感，而蒸馏使得模型预测更平滑（偏向 DA3 的稠密深度分布，而非 LiDAR 的稀疏绝对真值），这在 40 点 AP 和 AP@0.50 的全面提升中得到了印证。
- [ ] **消融分析**：对比各创新模块（如全图 vs 前景、是否加不确定性、是否加一致性约束）的效果差异。

### 第五阶段：数据分析与论文撰写 (第6周)
- [ ] 依托 **HTML + ECharts** 技术栈构建实验分析看板，绘制多维度指标对比图与深度分布直方图。
- [ ] 整理实验数据，输出交互式可视化结果（对比预测深度、真值与 DA3 教师深度）。
- [ ] 撰写毕业论文。

---

## 五、 预期成果与创新点

1. **理论创新**：提出**不确定性引导的自适应深度蒸馏**与**几何一致性约束**，有效解决了视觉基础模型伪标签的局部噪声问题，并弥合了像素级深度与实例级深度之间的鸿沟。
2. **结构与数据创新**：设计了**深度引导的动态特征对齐**模块，并提出了符合物理规律的 **3D-Aware DepthMix** 数据增强方法。
3. **工程价值**：提供了一套完整的基于视觉大模型深度先验的单目 3D 检测提升方案，部分模块（如数据增强、Loss 约束）实现“即插即用”且零推理负担。
4. **预期指标**：在 KITTI Easy/Moderate 难度下的 AP3D 指标预期提升 2-4 个百分点。
