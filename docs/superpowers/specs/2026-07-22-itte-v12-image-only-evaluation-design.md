# ITTE v1.2 纯图像主题迁移评估方案

## 1. 文档目的

ITTE（Icon Theme Transfer Evaluation）用于评价图标主题风格迁移是否成功。本方案只评价图像中可以观察和计算的视觉结果，不评价设计师的主观创作思想、文化隐喻、品牌策略或审美价值。

ITTE v1.2 的核心问题是：

> 给定一组参考 App 的 `original -> style_ref` 配对样例，目标 App 从 `target_original` 生成 `generated_icon` 后，是否完成了与参考样例相同类型、相近强度的视觉迁移，同时保留足够的目标图像结构，并形成一致、无明显缺陷的图标包。

该定义不要求预先把主题分类为手绘、3D、极简、黏土或其他风格。所有评分都以当前主题的参考图像为条件，因此可以评价未预先见过的主题类型。

## 2. 评估边界

### 2.1 ITTE 主分覆盖的内容

ITTE v1.2 主分只覆盖四类图像问题：

1. **Style Fidelity（风格忠实度）**：生成图的颜色、纹理、材质、描边、背景、构图和重复视觉 motif 是否接近当前主题参考图。
2. **Identity Preservation（视觉身份保持）**：生成图与目标原图之间是否保留可观测的主体结构、轮廓、局部区域和空间关系。
3. **Package Coherence（整包一致性）**：同一批生成图是否形成与参考主题包相近的内部视觉分布，是否存在明显离群图标。
4. **Visual Quality（视觉质量）**：生成图是否存在黑边、透明度异常、模糊、裁切、边缘破碎、背景污染和极端曝光等可检测缺陷。

### 2.2 ITTE 主分明确不覆盖的内容

以下内容不进入 ITTE 主分：

- 设计师的创作动机、设计哲学和情绪表达。
- 文化隐喻、叙事合理性和概念创意程度。
- “是否好看”“是否高级”等无参考审美判断。
- App 的商业语义是否表达准确。
- 品牌法务、商标合规和版权判断。
- 可读文字是否正确、OCR 是否通过以及文字合规。
- Prompt 与图片的文本语义对齐。
- Qwen、其他 VLM 或人工 QC 给出的历史分数。

因此，当一个目标 App 被重新设计成语义上合理、但与目标原图几乎没有视觉结构对应的新场景时，ITTE 会报告“视觉身份保持较弱”。这不等于宣判设计思想错误，只表示它超出了纯图像结构保持的评估边界。

### 2.3 用具体例子解释边界

| 实际情况 | ITTE 会怎样判断 | ITTE 不会怎样判断 |
|---|---|---|
| 生成图采用参考包的颜色、材质和描边，同时保留目标原图主体结构 | 风格分和身份分均可较高 | 不判断设计师为什么选择这些视觉元素 |
| 生成图语义上仍表示同一个 App，但把原 Logo 完全改成新的叙事场景 | 可能得到高风格分、低视觉身份分 | 不宣判新设计错误，也不判断用户是否仍能凭品牌知识认出 App |
| 生成图很好看，但视觉风格与参考主题不同 | 风格忠实度较低 | 不否定其独立审美价值 |
| 整包高度统一，但统一成了错误的另一种风格 | 包内离散度可较低，但到参考主题的距离较大，因此 Package Coherence 不能得到高分 | 不把“内部统一”误当成“迁移正确” |
| 原图有文字而生成图没有文字，或原图本来没有文字 | 不因文字存在与否直接奖惩 | 不执行 OCR、拼写或“保留文字/禁止文字”规则 |
| 生成图有乱码，但其他视觉属性正常 | ITTE 主分不处理文字正确性；产品需要时应由独立 OCR/文字合规门负责 | 不把 OCR 结果混入风格迁移分 |
| Prompt 与生成图语义一致 | 不影响 ITTE 主分 | 不使用 OpenCLIP 文图分证明风格迁移成功 |
| 生成阶段 Qwen QC 判定通过 | 只可记录为外部诊断 | 不提高 ITTE 分数，也不改变 ITTE 决策 |

所以，ITTE 的 `style_transfer_success` 只能解释为：

> 在给定参考图像证据下，该输出完成了可观测的视觉风格迁移，保留了足够的目标图像结构，整包分布一致，并且没有严重技术缺陷。

它不能被扩展解释为“设计思想正确”“商业上可发布”“品牌语义完全正确”或“所有人都会觉得好看”。

### 2.4 确定性要求

主评估必须满足：

- 相同输入文件、模型权重、依赖版本和配置重复运行，输出数值完全一致。
- 不在评估阶段调用 Qwen、Wan 或任何生成式 VLM。
- OpenCLIP 文图分不进入主分；v1.2 默认不再输出它也不影响决策。
- 生成阶段 Qwen QC 仅作为报告中的外部诊断信息，不参与任何主分、阈值或决定。
- 每个报告记录输入 SHA256、模型 revision、代码 commit、依赖版本和评分配置版本。

## 3. 输入定义

对某个主题，定义参考配对集合：

\[
\mathcal{R} = \{(O_i, S_i)\}_{i=1}^{N}
\]

其中：

- \(O_i\)：参考 App 原始图。
- \(S_i\)：同一个参考 App 的真实主题图 `style_ref`。
- \(N\)：完整参考配对数量，最低为 5，推荐为 8 至 12。

对待评价的目标 App，定义：

\[
\mathcal{G} = \{(T_j, G_j)\}_{j=1}^{M}
\]

其中：

- \(T_j\)：目标 App 原始图。
- \(G_j\)：生成后的主题图标。
- \(M\)：当前生成包中的目标 App 数量。

若评测集已经拥有目标 App 的真实主题图，记为 \(H_j\)。`hidden_gold` 仅用于 benchmark 和指标校准，不是线上 ITTE 运行的必需输入，也不要求生成图像素级复刻 \(H_j\)。

## 4. 统一预处理

所有指标使用同一套可复现预处理，避免当前不同服务以不同方式处理 Alpha、尺寸和背景。

每张图生成三个视图：

1. **Appearance View**：转换到 sRGB，在统一中性浅色背景上合成 Alpha，保持原始画布比例，用于风格和包级评价。
2. **Structure View**：保留 Alpha 和前景 mask；没有 Alpha 时使用边缘四角颜色估计背景，得到前景区域并归一化到固定方形画布，用于身份结构评价。
3. **Artifact View**：保留原始像素、Alpha、尺寸和边界，不做背景合成，用于质量缺陷检测。

标准尺度为 224 和 448 两级。224 用于全局表征和快速统计，448 用于局部 patch、描边和边界检查。所有 resize 使用固定插值方式并写入报告。

## 5. 总体评分结构

### 5.1 理论骨架与证据等级

ITTE 将每个组成部分标记为以下三个证据等级，报告和论文写作必须保留这个区别：

1. **A：直接采用（direct adoption）**：直接使用论文提出的表征或公开指标定义，例如 VGG Gram、LPIPS、DISTS 的结构/纹理分解和 KID/MMD。
2. **B：论文启发后改造（paper-inspired adaptation）**：使用论文证明有效的特征能力，但任务公式由 ITTE 新定义，例如基于 DINOv3 dense feature 的双向图标结构对应和重复 motif 检测。
3. **C：ITTE 工程假设（engineering hypothesis）**：权重、阈值、参考归一化公式、颜色/Alpha/裁切规则等由本项目提出，必须通过图标评测集校准，不能写成论文已经证明的结论。

四维结构的核心理论起点是 [Wright and Ommer, *ArtFID: Quantitative Evaluation of Neural Style Transfer*](https://arxiv.org/abs/2207.12280)。ArtFID 明确将神经风格迁移评价拆成内容保持和风格匹配，而不是只看单一相似度。ITTE 采用这一“双目标不可混为一谈”的思想，将其对应为 `Style Fidelity` 与 `Identity Preservation`，再针对图标包任务增加 `Package Coherence` 和可解释的 `Visual Quality`。

这里属于 **B 级改造**：ITTE 不直接复制 ArtFID 的具体乘法公式，也不在小型单包上复制其 FID 部分。新增的整包和质量维度是图标任务扩展，不是 ArtFID 原论文结论。

### 5.2 四维输出与总分

ITTE v1.2 首先输出四维向量，不允许只保留一个总分：

```text
Style Fidelity
Identity Preservation
Package Coherence
Visual Quality
```

在人工校准完成前使用以下先验权重：

\[
ITTE = 0.35 S_{style} + 0.30 S_{identity} + 0.20 S_{package} + 0.15 S_{quality}
\]

这些具体权重属于 **C 级 ITTE 工程先验**，不是论文结论。论文直接支持的是 ArtFID 的“风格匹配与内容保持应分别测量”；包一致性与技术质量则是本项目针对图标包增加的维度。最终权重必须在开发集上拟合，并在盲测集冻结验证。

## 6. Style Fidelity：风格忠实度

### 6.1 参考归一化思想

对任意风格表征 \(\phi\)，计算：

- \(D_{GR}\)：生成图到参考主题图中心的平均距离。
- \(D_{TR}\)：目标原图到参考主题图中心的平均距离。
- \(D_{RR}\)：参考主题图留一法内部距离。

相对迁移分为：

\[
S_{transfer}(\phi)=100\cdot clip\left(\frac{D_{TR}-D_{GR}}{D_{TR}-D_{RR}+\epsilon},0,1\right)
\]

该公式不要求所有主题共享一个绝对距离阈值。`0` 表示生成图没有比目标原图更接近主题；`100` 表示生成图进入了参考主题自身的离散范围。

**证据等级 C**：相对迁移公式是 ITTE 针对“不同主题尺度不可直接比较”提出的工程归一化，不是已有论文中的原公式。它受 ArtFID 条件化比较风格与内容的评价结构启发，但其有效性必须通过 13 节的人工视觉校准和跨主题盲测证明。

### 6.2 VGG Gram 风格分

从固定预训练 VGG 的多个层提取特征图 \(F^l\)，计算通道相关矩阵：

\[
Gram^l = F^l(F^l)^T
\]

使用多层 Gram 向量的余弦距离计算 `VGG Gram Transfer Score`。

**论文来源**：[Gatys, Ecker, Bethge, *A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576)。该论文使用深层特征表示内容、使用特征相关矩阵表示风格。ITTE 直接采用其 Gram 风格表征思想，但不执行风格生成优化，只用于参考条件式评价。

**证据等级 A/B**：Gram 表征本身为 A 级直接采用；将其变成参考归一化的图标迁移分为 B 级改造。

### 6.3 DISTS 纹理分量

DISTS 同时建模深层特征的结构相关和空间均值统计。ITTE 只将其中偏纹理的分量用于风格辅助，不把完整 DISTS 距离重复用于多个主分。

**论文来源**：[Ding et al., *Image Quality Assessment: Unifying Structure and Texture Similarity*](https://arxiv.org/abs/2004.07728)。该论文证明深层特征空间均值能够表达纹理外观，并将结构与纹理相似性统一到同一感知指标中。ITTE 借用其结构/纹理解耦思想。

**证据等级 A/B**：DISTS 分解思想为 A 级直接采用；仅抽取纹理分量并进行主题参考归一化为 B 级改造。

### 6.4 确定性视觉属性分

从 Appearance View 提取：

- RGB/HSV 颜色矩和直方图。
- 背景颜色、背景饱和度和背景面积。
- 边缘密度、边缘强度、方向直方图和暗色描边比例。
- 前景面积、包围盒、质心、留白和主体尺度。
- 灰度熵、颜色数量、Laplacian 能量和局部纹理能量。

每个属性分别计算参考归一化迁移分。只有当该属性在参考主题内部稳定、且 `original -> style_ref` 确实发生明显变化时才进入聚合。

**证据等级 C**：这部分不是对某一篇论文指标的直接复现，而是标准数字图像处理统计的工程组合。其理论角色是补足 Gram/DISTS 难以解释的颜色、背景和构图维度；所有属性必须通过可靠性门控，不能因重复测量颜色和边缘而被重复加权。

### 6.5 重复视觉 motif 分

使用 DINOv3 dense patch 特征，在多个 `style_ref` 中发现重复出现、但在对应 `original` 中不稳定出现的 patch 原型。随后计算这些原型在每张生成图前景中的覆盖率和空间集中度。

该指标只能判断“参考图中存在重复视觉局部，并且生成图也出现了相似局部”，不能命名该 motif，也不能判断它的设计含义。

**论文来源**：[Siméoni et al., *DINOv3*](https://arxiv.org/abs/2508.10104) 及其[官方 dense matching 示例](https://github.com/facebookresearch/dinov3)。DINOv3 提供高质量 dense feature，官方实现展示了跨图 patch matching。ITTE 将跨图局部对应能力改造成参考包重复视觉模式检测；这属于基于论文能力的任务适配，不是 DINOv3 论文原生提出的风格评分。

**证据等级 B**：DINOv3 特征来自论文，motif 原型发现、覆盖率和门控公式由 ITTE 定义。

### 6.6 风格子分聚合

校准前的先验组合为：

```text
40% VGG Gram transfer
20% DISTS texture transfer
25% deterministic visual attributes
15% DINOv3 repeated motif coverage
```

某个分量不可靠时不以零分惩罚，而是从分母移除并对剩余权重重新归一化。若有效权重低于 60%，该主题的风格评价标记为 `low_confidence`。

以上子权重和 60% 门槛均为 **C 级工程先验**，必须由校准集确认或修改。

## 7. Identity Preservation：视觉身份保持

### 7.1 边界定义

身份分只测量 \(T_j\) 和 \(G_j\) 的图像结构对应，不判断人是否能根据语义、文字或品牌知识认出 App。

为了允许主题本身对结构进行合理重构，身份相似度需要用参考配对 \((O_i,S_i)\) 估计当前主题通常允许的变形强度。不能用固定的像素相似阈值要求所有主题保留相同程度的原始 Logo。

### 7.2 DINOv3 dense correspondence

分别提取目标原图和生成图的前景 patch 特征，计算双向最近邻匹配：

\[
M_{T\rightarrow G}=\frac{1}{|P_T|}\sum_{p\in P_T}\max_{q\in P_G}cos(p,q)
\]

\[
M_{G\rightarrow T}=\frac{1}{|P_G|}\sum_{q\in P_G}\max_{p\in P_T}cos(q,p)
\]

再结合匹配 patch 的空间位置一致性，得到 dense identity similarity。双向匹配防止生成图只保留一个很小的局部就获得高分。

**论文来源**：DINOv3 的 dense feature 与跨图 patch matching 能力，来源同 6.5。ITTE 不再以全图 CLS 最近邻作为主要身份分，因为 CLS 会同时受到主题背景、颜色和共享 motif 影响。

**证据等级 B**：dense feature 为论文能力，双向最近邻、空间一致性和图标前景门控为 ITTE 任务公式。

### 7.3 DISTS 结构分量

在 Structure View 上使用 DISTS 的结构分量，评价多尺度主体结构是否保持，同时允许轻微平移、缩放和纹理替换。

**论文来源**：DISTS 论文。ITTE 只把结构分量用于身份，把纹理分量用于风格，避免同一个完整距离在两个维度中重复计权。

**证据等级 A/B**：结构分量来自 DISTS；在 Structure View 上单独用于图标身份为 ITTE 改造。

### 7.4 LPIPS 感知内容辅助

在前景归一化视图上计算 LPIPS，并使用参考配对 \((O_i,S_i)\) 的 LPIPS 分布进行主题内归一化。它用于检测整体感知变化是否远大于该主题正常迁移强度，不单独决定身份成功。

**论文来源**：[Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html)。该论文使用人类感知判断数据验证深层特征距离比传统像素指标更符合视觉相似性。ArtFID 也使用 LPIPS 表达风格迁移中的内容保持。ITTE 对其做参考主题归一化，避免把预期的风格变化全部误判为内容损失。

**证据等级 A/B**：LPIPS 距离为 A 级直接采用；使用 `original -> style_ref` 分布归一化为 B 级改造。

### 7.5 轮廓和边缘硬诊断

计算前景面积比例、包围盒、质心、轮廓距离和边缘覆盖率。这些指标主要用于解释失败和触发严重结构异常门槛，不直接以大权重加入连续身份分。

**证据等级 C**：这是传统几何图像比较的工程诊断，用于捕获深层特征可能忽略的裁切、主体消失和轮廓崩坏。

### 7.6 主题自适应身份基线

对每个参考配对计算 \(M(O_i,S_i)\)，形成“该主题正常迁移后仍保留多少结构”的正样本分布。负样本基线来自冻结并版本化的其他 App 原图干扰库，而不是当前包中临时出现的 App，避免相同生成图因包内成员变化而改变身份分。

校准前的身份组合为：

```text
50% DINOv3 bidirectional dense correspondence
30% DISTS structure similarity
20% reference-normalized LPIPS content similarity
```

轮廓和边缘异常作为 hard diagnostic。最终权重由人工图像结构判断校准。

身份子权重、正负基线映射和 hard diagnostic 阈值均为 **C 级工程先验**；它们不是 DINOv3、DISTS 或 LPIPS 论文给出的图标阈值。

## 8. Package Coherence：整包一致性

### 8.1 包内一致性

对每张生成图构建只包含风格信息的联合向量，计算：

- 包内两两距离的 median、P90 和 MAD。
- 每张图到生成包 robust centroid/medoid 的距离。
- 每个风格属性的包内方差。
- 最差 App、P10 分数和离群 App 数量。

离群阈值使用 median + MAD，而不是均值 + 标准差，降低极端样本对阈值本身的影响。

**证据等级 C**：median、MAD 和 medoid 是稳健统计工具，但这里的联合特征、离群公式和阈值是 ITTE 的包级工程定义，不声称来自某篇风格迁移论文。

### 8.2 参考包归一化

生成包内部统一并不代表它属于指定主题，因此还需要比较：

- 生成包内部离散度与参考 `style_ref` 包内部离散度。
- 每张生成图到参考主题 medoid/centroid 的距离。
- 生成包与参考包各风格子空间的分布差异。

Package Coherence 同时满足“生成包内部统一”和“生成包位于参考主题附近”才可获得高分。

**证据等级 B/C**：比较生成与参考特征分布受 KID/MMD 两样本检验思想启发；单包的 robust centroid、离散度比值和聚合公式由 ITTE 定义。

### 8.3 KID/MMD 诊断

当同一主题累计不少于 20 张独立生成图和 20 张参考/真实主题图时，计算基于固定视觉特征的 MMD/KID，并使用 bootstrap 输出置信区间。单个只有 6 至 10 张图的包不使用 KID 决定成败。

**论文来源**：[Bińkowski et al., *Demystifying MMD GANs*](https://arxiv.org/abs/1801.01401)。该论文提出 Kernel Inception Distance，使用 MMD 的无偏估计比较真实和生成特征分布。ITTE 只在样本量满足最低要求时把它作为跨运行包级诊断。

**证据等级 A/B**：无偏 MMD 估计为 A 级采用；特征选择、20 张最低启用量和跨运行聚合方式为待校准的 B/C 级设置。`20` 不是论文保证充分的通用样本量，只是禁止在更小单包上做主判定的保守工程下限。

### 8.4 为什么不直接使用 FID

FID 需要估计高维特征的均值和协方差，小型图标包样本远不足以稳定估计。有限样本 FID 具有模型相关偏差，固定相同样本数量也不能消除。

**论文依据**：[Chong, Forsyth, *Effectively Unbiased FID and Inception Score and where to find them*](https://arxiv.org/abs/1911.07023)。因此 ITTE v1.2 不把传统 FID 用于单包主分。

## 9. Visual Quality：视觉质量

视觉质量只检测技术缺陷，不评价审美。每张图检查：

- Alpha 最小值、最大值、边界 Alpha 和异常半透明面积。
- 四边和四角背景污染、黑边与非主题色边框。
- 前景是否接触画布边界，是否存在裁切。
- Laplacian/梯度能量是否异常低，检测模糊或空图。
- 边缘连通分量是否异常增加，检测边缘破碎和噪声。
- 亮度、对比度、饱和度是否严重超出参考主题 robust envelope。
- 前景面积、主体尺度和留白是否严重超出参考主题范围。
- 分辨率、长宽比、颜色模式和文件可解码性。

一般质量属性以参考主题 median/MAD 归一化；文件损坏、严重裁切、明显黑边和大面积异常透明属于主题无关 hard failure。

**证据等级 C**：该部分是可解释的确定性图像处理规则，不直接套用 BRISQUE/NIQE。BRISQUE/NIQE 基于自然场景统计，而图标属于高度合成、扁平或透明的非自然图像域，未经图标域人工校准不进入主分。每项缺陷阈值都必须由受控缺陷样本和人工标注校准。

## 10. 决策规则与置信度

### 10.1 校准前的临时等级

```text
>= 80: strong
70-79.99: acceptable
60-69.99: weak
< 60: failed
```

这些等级只用于开发期可读输出，不能写成论文结论。正式阈值由校准集确定。

四档分界和下面所有硬门槛均属于 **C 级工程先验**，没有任何一篇所引论文证明 `80` 或 `35` 是通用成功线。

### 10.2 硬门槛

即使总分较高，出现以下任一情况也不能判为 `style_transfer_success`：

- 任一 App 的身份分低于 35。
- 包级身份分 P10 低于 45。
- 任一 App 出现文件损坏、严重裁切、大面积黑边或异常透明。
- 包内出现严重风格离群，且该 App 到参考主题的距离也超过参考最大离散范围。
- 有效风格指标权重低于 60%，即参考证据不足。

阈值在正式校准后冻结到版本化配置；修改阈值必须提升 ITTE 版本。

### 10.3 评估置信度

置信度由参考数量、参考内部一致性、可用指标比例和分母稳定性共同决定：

```text
high: 至少 8 对参考，参考一致，至少 80% 风格权重有效
medium: 至少 5 对参考，至少 60% 风格权重有效
low: 参考少于 5 对、参考本身不一致或有效权重低于 60%
```

`low` 时仍可输出诊断数据，但 decision 必须为 `insufficient_reference_evidence`，不能给出正式通过结论。

## 11. 三主题评测集构建

### 11.1 固定留出真值

每个主题把已有 `original -> style_ref` 配对分成三组。以每主题 12 对为例：

```text
6 reference apps
3 development target apps
3 blind test target apps
```

生成 development/test target 时，只向生成系统提供 `reference apps` 的完整配对和目标 App 的 `original`。目标 App 已有的真实主题图作为 `hidden_gold`，在生成完成前不得进入 Qwen、Wan、QC 或任何输入文件。

`hidden_gold` 只用于：

- 人工对照。
- 校准自动指标。
- 判断生成结果是否达到已知合格主题图的大致视觉水平。

它不是像素级唯一答案。

### 11.2 生成重复次数

每个目标 App 进行 4 次独立生成，每次保留 3 张候选：

```text
每目标 12 张 candidates
每目标 4 张 generation-QC selected
每目标 4 张 human oracle best-of-3
```

若三个主题各有 6 个 dev/test target，则得到：

```text
3 themes × 6 targets × 4 runs × 3 candidates = 216 candidates
3 themes × 6 targets × 4 runs = 72 selected outputs
```

评测必须分别报告：

- `all_candidates`：生成模型总体输出分布。
- `generation_selected`：当前生成系统加 Qwen QC 的实际表现。
- `human_oracle_best_of_3`：候选池能力上限，用于判断问题来自生成还是选择。

### 11.3 包级样本

同一主题、同一 run、同一 split 的多个目标输出组成一个自然包。不同 run 的图片不得混合成自然包。为了校准离群检测，可额外构造受控包：向正常包替换一张真实的异主题、异常背景或异常线条图标，并明确标记 `controlled_corruption=true`。受控包只用于指标校准，不计入生成模型正式成绩。

## 12. 人工校准只评价图像

每张生成图由至少 3 名标注者独立判断：

1. 与主题参考图相比，风格视觉上有多接近，1 至 5 分。
2. 与目标原图相比，主体视觉结构保留程度，1 至 5 分。
3. 是否存在可见图像缺陷，1 至 5 分。
4. 在两张候选中，哪张风格更接近参考、哪张结构保留更好。

每个包判断：

1. 包内是否使用统一视觉语言，1 至 5 分。
2. 哪些 App 是视觉离群项。
3. 是否存在会阻止整包通过的严重图像问题。

标注问题不得包含“是否有创意”“是否符合设计师意图”“品牌含义是否合理”等主观设计问题。

## 13. 校准与盲测

### 13.1 数据切分

- development：拟合单指标单调映射、主分权重和通过阈值。
- validation：选择配置，不能继续直接拟合。
- blind test：配置冻结后只运行一次，不得根据结果回调阈值。

三个主题还需要执行 Leave-One-Theme-Out：使用两个主题校准，在第三个未参与校准的主题上测试，轮换三次。它不能证明覆盖全部未来主题，但能检查指标是否只记住某个主题的颜色和构图。

### 13.2 校准目标

- 四个自动维度与对应人工图像评分的 Spearman 相关性。
- 候选成对排序的 Kendall 相关性/排序准确率。
- 严重身份丢失、包内离群和质量缺陷的 Recall、Precision、F1。
- 错误判为完全成功的比例。
- bootstrap 95% 置信区间。

第一版验收标准：

```text
总体 Spearman >= 0.75
每个主维度 Spearman >= 0.65
严重身份丢失 Recall >= 0.90
严重质量缺陷 Recall >= 0.90
完全成功误通过率 <= 0.05
相同输入重复运行结果和报告哈希一致
```

## 14. 报告结构

ITTE v1.2 报告至少包含：

```json
{
  "itte_version": "v1.2-image-only",
  "evaluation_scope": "observable_image_transfer_only",
  "itte_score": 0,
  "style_fidelity": {},
  "identity_preservation": {},
  "package_coherence": {},
  "visual_quality": {},
  "hard_failures": [],
  "evaluation_confidence": "high | medium | low",
  "decision": "",
  "per_app": [],
  "diagnostics": {
    "strict_delta": {},
    "generation_qwen_qc": {},
    "controlled_corruption": false
  },
  "provenance": {
    "input_sha256": {},
    "model_revisions": {},
    "code_commit": "",
    "dependency_versions": {},
    "scoring_config_hash": ""
  }
}
```

Qwen QC 可以出现在 `diagnostics`，但不得参与任何主分或 decision。

## 15. 论文思想映射表

| ITTE 组件 | 证据等级 | 论文思想 | 使用方式 | 与原论文的差异 |
|---|---|---|---|---|
| 四维评价骨架 | B | ArtFID 2022 | 风格匹配与内容保持必须分开报告 | 不复制 ArtFID 总分；扩展整包一致性和质量维度 |
| VGG Gram style | A/B | Gatys et al. 2015 | 多层特征相关矩阵表达风格 | 只评价，不执行生成优化；增加参考归一化 |
| DISTS texture | A/B | Ding et al. 2020 | 深层空间均值表达纹理外观 | 只进入风格分并做主题归一化 |
| DISTS structure | A/B | Ding et al. 2020 | 深层特征相关表达结构相似 | 只进入身份分，使用前景视图 |
| LPIPS content | A/B | Zhang et al. 2018；ArtFID 2022 | 深层特征距离表达感知变化与内容保持 | 使用参考 `original -> style_ref` 归一化，不直接使用绝对距离 |
| DINOv3 dense identity | B | Siméoni et al. 2025 | dense patch feature 和跨图局部匹配 | 改造为图标主体双向对应和空间一致性 |
| DINOv3 motif | B | Siméoni et al. 2025 | 多图 patch matching | 改造为参考包重复局部模式覆盖率 |
| KID/MMD package diagnostic | A/B | Bińkowski et al. 2018 | 以无偏 MMD 估计比较特征分布 | 只在跨运行样本量足够时启用 |
| 禁止小包 FID 主判定 | A | Chong, Forsyth 2019 | 有限样本 FID 存在模型相关偏差 | ITTE 单包不使用传统 FID |
| 相对迁移公式 | C | ArtFID 仅提供双目标启发 | 按当前主题参考离散度归一化 | ITTE 新公式，必须跨主题校准 |
| 先验权重和成功阈值 | C | 无直接论文结论 | 开发期聚合与决策 | 必须由评测集拟合，不能当作理论常数 |
| 颜色、背景、边缘、Alpha、裁切 | C | 标准数字图像处理 | 可解释属性和 artifact 检测 | 工程指标，必须经过图标域人工校准 |

## 16. 参考文献

1. Gatys, L. A., Ecker, A. S., Bethge, M. [*A Neural Algorithm of Artistic Style*](https://arxiv.org/abs/1508.06576). arXiv:1508.06576, 2015.
2. Wright, M., Ommer, B. [*ArtFID: Quantitative Evaluation of Neural Style Transfer*](https://arxiv.org/abs/2207.12280). arXiv:2207.12280, 2022.
3. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., Wang, O. [*The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html). CVPR, 2018.
4. Ding, K., Ma, K., Wang, S., Simoncelli, E. P. [*Image Quality Assessment: Unifying Structure and Texture Similarity*](https://arxiv.org/abs/2004.07728). arXiv:2004.07728, 2020.
5. Siméoni, O. et al. [*DINOv3*](https://arxiv.org/abs/2508.10104). arXiv:2508.10104, 2025.
6. Bińkowski, M., Sutherland, D. J., Arbel, M., Gretton, A. [*Demystifying MMD GANs*](https://arxiv.org/abs/1801.01401). ICLR, 2018.
7. Chong, M. J., Forsyth, D. [*Effectively Unbiased FID and Inception Score and where to find them*](https://arxiv.org/abs/1911.07023). arXiv:1911.07023, 2019.

## 17. 最终定义

ITTE v1.2 是一个参考条件式、纯图像、确定性的图标主题迁移评价框架。它评价的是视觉风格变化、视觉结构保持、生成集合一致性和技术质量，不评价设计思想和语义创意。它不承诺覆盖所有审美和设计意图，但能够在不预定义主题类别的前提下，对任意具有足够 `original -> style_ref` 配对证据的主题执行同一套可复现视觉测量。
