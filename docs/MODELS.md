# Model weights

Large model weights are local reproducible caches and are not committed to Git.

## DINOv3

- Model ID: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Source used by the current configuration: ModelScope
- Weight file SHA-256 verified on this machine: `9A21AC3DF0C63839D62612DDA6F454D816C25611CC7A52966ED5A5A94921DC8B`
- Expected local cache root: `models/modelscope/`

The evaluation code resolves/downloads the configured snapshot through `modelscope` when the cache is absent.

## VGG16

- TorchVision weight enum: `VGG16_Weights.IMAGENET1K_V1`
- Official checkpoint SHA-256 verified on this machine: `397923AF8E79CDBB6A7127F12361ACD7A2F83E06B05044DDF496E83DE57A5BF0`
- Expected local cache root: `models/torch/`

TorchVision downloads the checkpoint into `TORCH_HOME` when absent.

## DISTS and LPIPS

`DISTS-pytorch==0.1` and `lpips==0.1.4` provide their published weights through the installed Python packages. They are retained as style/diagnostic backends; their identity outputs do not enter the ITTE v1.3 primary identity score.

## Reproducibility

Install CUDA PyTorch first, then the remaining dependencies, so a generic dependency install cannot replace the CUDA build:

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
```

Run `benchmark/tools/smoke_itte_device.py` to populate and verify model caches on either `cuda:0` or `cpu`.
