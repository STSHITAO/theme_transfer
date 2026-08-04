# Theme Transfer

一个面向移动 App 图标的主题风格迁移与纯图像评测项目。生成链路从真实主题示例中分析共享设计规则，通过 Qwen 规划、Wan 生图和候选 QC 生成整包图标；ITTE 独立评估可观察的风格迁移、身份保持、整包一致性和技术缺陷。

ITTE = Icon Theme Transfer Evaluation。v1.3 的四个主维度是 Style Fidelity、Identity Preservation、Package Coherence 和 Visual Quality。风格侧包含 VGG16 多层 Gram 等纯图像证据；Package Coherence 同时衡量整包统一性与参考主题归属，因此整包内部统一但远离参考主题，不能得到高分。评测只输出诊断与报告，不再生成 `generation_feedback_prompt.md`，也不把评测文字反馈接回生图链路。

## 项目状态

- 生成工作流：已实现。
- ITTE v1.3：已在真实设计师 Benchmark 上完成五折验证。身份主分使用 DINOv3 同标签检索百分位；DISTS/LPIPS 保留为诊断，不进入身份主分。
- Benchmark：`evaluation_set_v1` 已通过独立完整性校验。
- GPU：使用 Conda 环境 `pytorch`，支持 CUDA 与磁盘特征缓存；保留 CPU 接口。
- Web UI / 部署：尚未实现。

进度与待办见 [docs/PROGRESS.md](docs/PROGRESS.md)，产品和技术说明见 [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) 与 [docs/TECH_DESIGN.md](docs/TECH_DESIGN.md)。

## 环境

当前机器推荐直接使用已配置的 GPU 环境：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
```

CPU 兼容运行：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cpu'
$env:TPQS_BATCH_SIZE='1'
```

如需新建环境，先按照本机驱动安装匹配的 CUDA PyTorch/TorchVision，再安装 `requirements.txt`，避免通用安装覆盖 GPU 构建。

## 生成主题包

在 `.env` 中配置本地 API 参数后运行：

```powershell
conda activate pytorch
python backend/run_package.py
```

输入位于：

```text
data/styles/<theme_id>/<reference_app>/
data/targets/<target_app>/
```

输出位于 `data/packages/<package_id>/`。`.env` 包含密钥，严禁提交。

## 运行已有包的 ITTE

在 `evaluation/evaluate_package.py` 中设置主题、包和评测 ID，然后运行：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
python evaluation/evaluate_package.py
```

## 验证当前 ITTE 方案

真实 Benchmark 位于 [benchmark/evaluation_set_v1](benchmark/evaluation_set_v1)。当前版本基线 Runner 不修改 ITTE 算法：

```powershell
conda activate pytorch
$env:TPQS_DEVICE='cuda:0'
$env:TPQS_BATCH_SIZE='2'
python benchmark/tools/evaluate_current_itte.py --references 8 --queries 4
```

Runner 使用身份不重叠的参考/查询划分，对比设计师真实主题图与未迁移原图控制，并把报告写入 Benchmark 的 `results/`。冻结 v1.2、最终 v1.3 以及逐折对照分别位于：

- `benchmark/evaluation_set_v1/results/itte_v12_baseline_gpu_cv/`
- `benchmark/evaluation_set_v1/results/itte_v13_final_gpu_cv/`
- `benchmark/evaluation_set_v1/results/itte_v13_final_gpu_cv/comparison_to_v12/`

真实标签身份检索证据位于 `identity_retrieval_dinov3/` 与 `identity_retrieval_perceptual/`。

设备接口冒烟：

```powershell
python benchmark/tools/smoke_itte_device.py --device cuda:0
python benchmark/tools/smoke_itte_device.py --device cpu
```

## 测试

```powershell
conda activate pytorch
python -m pytest -q
python benchmark/tools/validate_evaluation_set.py
```

## 重要边界

- ITTE 主分只使用图像证据。
- OpenCLIP、Prompt 文图匹配和生成阶段 Qwen QC 不进入主分。
- Benchmark 主题描述只用于解释，不参与评分。
- 修改指标前必须保存相同数据划分下的原版本基线，修改后使用同一划分复测。
- 模型缓存位于 `models/`，特征缓存位于 `data/evaluations/_cache/`；不要提交临时或不完整下载文件。
