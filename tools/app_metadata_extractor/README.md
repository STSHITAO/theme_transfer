# 独立应用元数据提取工具

该工具只读取爬虫目录中的应用市场描述，并用 Qwen 提取 `category` 和 `core_function`。它不导入项目 `backend`，不修改 `dataset`，不触发生图或 ITTE，也不会读取或发送 `主题图标.png`。

输入目录格式：

```text
应用描述/
├── app_ids.json
├── 应用 A/
│   ├── 应用描述.txt
│   └── 主题图标.png  # 工具忽略
└── 应用 B/
    └── 应用描述.txt
```

`app_ids.json` 由人维护，应用改名时保持已有 ID 不变：

```json
{
  "apps": {
    "应用 A": "stable_app_a",
    "应用 B": "stable_app_b"
  }
}
```

先做纯本地校验：

```powershell
conda run -n pytorch python tools/app_metadata_extractor/extract.py --input-dir 应用描述 --validate-only
```

不调用 API，验证完整写入流程：

```powershell
conda run -n pytorch python tools/app_metadata_extractor/extract.py --input-dir 应用描述 --mock --force
```

使用根目录 `.env` 中的 Qwen 配置执行真实提取：

```powershell
conda run -n pytorch python tools/app_metadata_extractor/extract.py --input-dir 应用描述 --force
```

默认结果写到本工具的 `output/apps.generated.json`，同时生成审阅报告。结果不会自动进入项目主数据；必须人工核对后再决定是否另行导入。
