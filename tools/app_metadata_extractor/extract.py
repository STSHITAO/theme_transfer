"""从独立爬虫目录提取 App 类别与核心功能，不接入项目主流程。"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
from dotenv import load_dotenv


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path("应用描述")
DEFAULT_OUTPUT = TOOL_DIR / "output" / "apps.generated.json"
DEFAULT_PROMPT = TOOL_DIR / "prompt.md"
DESCRIPTION_FILE = "应用描述.txt"
ID_MAP_FILE = "app_ids.json"
OUTPUT_FIELDS = {"app", "category", "core_function"}


class ExtractionError(ValueError):
    """输入、模型输出或断点文件不满足独立工具约束。"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--id-map", type=Path, help="默认读取 <input-dir>/app_ids.json。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--app-id", action="append", help="只提取指定稳定 ID；可重复。")
    parser.add_argument("--force", action="store_true", help="重新提取已有断点结果。")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="只校验爬虫目录，不调用模型。")
    parser.add_argument("--mock", action="store_true", help="不调用模型，用本地占位值验证完整写入流程。")
    return parser


def load_crawler_directory(input_dir: Path, id_map_path: Path | None = None) -> dict[str, dict[str, str]]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ExtractionError(f"爬虫输入目录不存在：{input_dir}")
    mapping_path = (id_map_path or input_dir / ID_MAP_FILE).resolve()
    mapping_payload = _load_json(mapping_path, "App ID 映射")
    if not isinstance(mapping_payload, dict) or set(mapping_payload) != {"apps"}:
        raise ExtractionError("App ID 映射顶层必须严格为 {'apps': {...}}。")
    mapping = mapping_payload["apps"]
    if not isinstance(mapping, dict) or not mapping:
        raise ExtractionError("App ID 映射中的 apps 必须是非空对象。")

    folders = sorted(
        (path for path in input_dir.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    )
    folder_names = {path.name for path in folders}
    missing_mappings = sorted(folder_names - set(mapping), key=str.casefold)
    extra_mappings = sorted(set(mapping) - folder_names, key=str.casefold)
    if missing_mappings:
        raise ExtractionError("以下爬虫目录缺少稳定 App ID：" + "、".join(missing_mappings))
    if extra_mappings:
        raise ExtractionError("ID 映射包含不存在的爬虫目录：" + "、".join(extra_mappings))
    if not folders:
        raise ExtractionError("爬虫输入目录中没有应用子目录。")

    result: dict[str, dict[str, str]] = {}
    seen_ids: dict[str, str] = {}
    for folder in folders:
        app_id = mapping[folder.name]
        if not isinstance(app_id, str):
            raise ExtractionError(f"{folder.name!r} 的 App ID 必须是字符串。")
        app_id = app_id.strip()
        _validate_identifier(app_id)
        folded = app_id.casefold()
        if folded in seen_ids:
            raise ExtractionError(
                f"稳定 App ID 重复：{app_id!r} 同时用于 {seen_ids[folded]!r} 和 {folder.name!r}。"
            )
        description_path = folder / DESCRIPTION_FILE
        description = _read_required_text(description_path)
        seen_ids[folded] = folder.name
        result[app_id] = {
            "app": app_id,
            "display_name": folder.name,
            "store_description": description,
            "source_description_file": str(description_path),
        }
    return {app_id: result[app_id] for app_id in sorted(result, key=str.casefold)}


def extract_directory(
    input_dir: Path,
    output_path: Path,
    *,
    id_map_path: Path | None = None,
    prompt_path: Path = DEFAULT_PROMPT,
    batch_size: int = 10,
    selected_app_ids: list[str] | None = None,
    force: bool = False,
    continue_on_error: bool = False,
    mock: bool = False,
) -> dict:
    if batch_size < 1:
        raise ExtractionError("batch_size 必须至少为 1。")
    source_apps = load_crawler_directory(input_dir, id_map_path)
    all_source_ids = set(source_apps)
    if selected_app_ids:
        requested = sorted(set(selected_app_ids), key=str.casefold)
        unknown = sorted(set(requested) - all_source_ids, key=str.casefold)
        if unknown:
            raise ExtractionError("未知的 --app-id：" + "、".join(unknown))
        source_apps = {app_id: source_apps[app_id] for app_id in requested}

    prompt = _read_required_text(prompt_path.resolve())
    output_path = output_path.resolve()
    completed = {
        app_id: profile
        for app_id, profile in _load_existing_output(output_path).items()
        if app_id in all_source_ids
    }
    resumed: list[str] = []
    pending: list[dict[str, str]] = []
    for app_id, source in source_apps.items():
        old = completed.get(app_id)
        if not force and _matches_source(old, source):
            resumed.append(app_id)
        else:
            completed.pop(app_id, None)
            pending.append(source)

    generated: list[str] = []
    failures: list[dict] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        try:
            extracted = _mock_extract(batch) if mock else extract_batch_with_qwen(batch, prompt)
            for item in extracted:
                source = source_apps[item["app"]]
                completed[item["app"]] = {
                    "app": item["app"],
                    "display_name": source["display_name"],
                    "category": item["category"],
                    "store_description": source["store_description"],
                    "core_function": item["core_function"],
                    "source_description_file": source["source_description_file"],
                }
                generated.append(item["app"])
            _write_checkpoint(output_path, completed, input_dir.resolve())
        except Exception as exc:
            failure = {
                "apps": [item["app"] for item in batch],
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            if not continue_on_error:
                report = _write_report(output_path, input_dir, generated, resumed, failures, mock)
                raise ExtractionError(f"批次提取失败，详情见：{report}") from exc

    if completed:
        _write_checkpoint(output_path, completed, input_dir.resolve())
    report_path = _write_report(output_path, input_dir, generated, resumed, failures, mock)
    return {
        "input_dir": str(input_dir.resolve()),
        "output": str(output_path),
        "report": str(report_path),
        "requested_app_count": len(source_apps),
        "generated_app_count": len(generated),
        "resumed_app_count": len(resumed),
        "failed_batch_count": len(failures),
        "failed_apps": [app for failure in failures for app in failure["apps"]],
        "mock": mock,
    }


def extract_batch_with_qwen(apps: list[dict[str, str]], prompt: str) -> list[dict[str, str]]:
    base_url = os.getenv("ALI_PLAN_BASE_URL")
    model = os.getenv("ALI_PLAN_MODEL")
    api_key = os.getenv("ALI_PLAN_API_KEY")
    missing = [
        key
        for key, value in (
            ("ALI_PLAN_BASE_URL", base_url),
            ("ALI_PLAN_MODEL", model),
            ("ALI_PLAN_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise ExtractionError("缺少 Qwen 配置：" + "、".join(missing))

    model_inputs = [
        {
            "app": item["app"],
            "display_name": item["display_name"],
            "store_description": item["store_description"],
        }
        for item in apps
    ]
    content = [
        {
            "text": (
                f"{prompt}\n\n【输入应用 INPUT_APPS】\n"
                f"{json.dumps({'apps': model_inputs}, ensure_ascii=False, indent=2)}"
            )
        }
    ]
    import dashscope
    from dashscope import MultiModalConversation

    dashscope.base_http_api_url = base_url
    last_error = None
    for _ in range(2):
        try:
            response = MultiModalConversation.call(
                model=model,
                api_key=api_key,
                messages=[{"role": "user", "content": content}],
                enable_thinking=False,
                timeout=120,
            )
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
    else:
        raise last_error
    status_code = _response_value(response, "status_code")
    if status_code and status_code != 200:
        raise ExtractionError(f"Qwen API 调用失败：{_safe_response(response)}")
    response_text = _extract_response_text(response)
    if not response_text:
        raise ExtractionError(f"Qwen 返回空响应：{_safe_response(response)}")
    return parse_qwen_output(response_text, model_inputs)


def parse_qwen_output(text: str, source_apps: list[dict[str, str]]) -> list[dict[str, str]]:
    parsed = _parse_json(text)
    if not isinstance(parsed, dict) or set(parsed) != {"apps"} or not isinstance(parsed["apps"], list):
        raise ExtractionError("Qwen 响应必须严格符合 {'apps': [...]}。")
    expected_ids = [item["app"] for item in source_apps]
    expected_set = set(expected_ids)
    extracted: dict[str, dict[str, str]] = {}
    for index, item in enumerate(parsed["apps"]):
        if not isinstance(item, dict) or set(item) != OUTPUT_FIELDS:
            raise ExtractionError(f"Qwen 第 {index} 项只能包含 app、category、core_function。")
        app_id = item.get("app")
        if app_id not in expected_set:
            raise ExtractionError(f"Qwen 返回未知 App ID：{app_id!r}。")
        if app_id in extracted:
            raise ExtractionError(f"Qwen 返回重复 App ID：{app_id!r}。")
        category = item.get("category")
        core_function = item.get("core_function")
        if not isinstance(category, str) or not category.strip() or "/" not in category:
            raise ExtractionError(f"{app_id!r} 的 category 必须是“一级分类 / 二级分类”。")
        if not isinstance(core_function, str) or not core_function.strip():
            raise ExtractionError(f"{app_id!r} 的 core_function 不能为空。")
        if len(category) > 80 or len(core_function) > 300:
            raise ExtractionError(f"{app_id!r} 的提取结果超过长度限制。")
        extracted[app_id] = {
            "app": app_id,
            "category": category.strip(),
            "core_function": core_function.strip(),
        }
    missing = [app_id for app_id in expected_ids if app_id not in extracted]
    if missing:
        raise ExtractionError("Qwen 遗漏 App ID：" + "、".join(missing))
    return [extracted[app_id] for app_id in expected_ids]


def _mock_extract(apps: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"app": item["app"], "category": "其他 / 待分类", "core_function": item["store_description"][:300]}
        for item in apps
    ]


def _validate_identifier(value: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ExtractionError(f"不安全的 App ID：{value!r}")


def _load_json(path: Path, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except FileNotFoundError as exc:
        raise ExtractionError(f"缺少{label}文件：{path}") from exc
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{label}不是有效 UTF-8：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"无法解析{label}（第 {exc.lineno} 行，第 {exc.colno} 列）：{exc.msg}") from exc


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError(f"JSON 键重复：{key!r}")
        result[key] = value
    return result


def _read_required_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ExtractionError(f"缺少文本文件：{path}") from exc
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"文本文件不是有效 UTF-8：{path}") from exc
    if not text:
        raise ExtractionError(f"文本文件为空：{path}")
    return text


def _load_existing_output(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = _load_json(path, "已有输出")
    if not isinstance(payload, dict) or not isinstance(payload.get("apps"), dict):
        raise ExtractionError("已有输出必须包含顶层 apps 对象。")
    return payload["apps"]


def _matches_source(profile, source) -> bool:
    return bool(
        isinstance(profile, dict)
        and profile.get("app") == source["app"]
        and profile.get("display_name") == source["display_name"]
        and profile.get("store_description") == source["store_description"]
        and isinstance(profile.get("category"), str)
        and profile["category"].strip()
        and isinstance(profile.get("core_function"), str)
        and profile["core_function"].strip()
    )


def _write_checkpoint(path: Path, apps: dict[str, dict], input_dir: Path) -> None:
    payload = {
        "schema_version": "standalone-app-metadata-extractor-v1",
        "source_directory": str(input_dir),
        "apps": {app_id: apps[app_id] for app_id in sorted(apps, key=str.casefold)},
    }
    _atomic_write_json(path, payload)


def _write_report(path: Path, input_dir: Path, generated, resumed, failures, mock) -> Path:
    report_path = path.with_name(path.stem + ".report.json")
    payload = {
        "schema_version": "standalone-app-metadata-extractor-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir.resolve()),
        "output": str(path.resolve()),
        "model": "mock" if mock else os.getenv("ALI_PLAN_MODEL", ""),
        "generated_apps": sorted(generated, key=str.casefold),
        "resumed_apps": sorted(resumed, key=str.casefold),
        "failures": failures,
        "human_review_required": True,
        "integration_note": "此结果未接入 dataset、生图或评测流程；如需使用，必须人工审核并另行导入。",
    }
    _atomic_write_json(report_path, payload)
    return report_path


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError:
        return None


def _extract_response_text(response) -> str:
    if isinstance(response, dict):
        content = response.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", [])
    else:
        output = getattr(response, "output", {})
        content = output.get("choices", [{}])[0].get("message", {}).get("content", [])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item["text"] for item in content if isinstance(item, dict) and "text" in item)
    return ""


def _response_value(response, key):
    return response.get(key) if isinstance(response, dict) else getattr(response, key, None)


def _safe_response(response) -> str:
    api_key = os.getenv("ALI_PLAN_API_KEY", "")
    return str(response).replace(api_key, "***") if api_key else str(response)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    id_map = args.id_map or args.input_dir / ID_MAP_FILE
    source_apps = load_crawler_directory(args.input_dir, id_map)
    if args.app_id:
        unknown = sorted(set(args.app_id) - set(source_apps), key=str.casefold)
        if unknown:
            raise SystemExit("未知的 --app-id：" + "、".join(unknown))
    if args.validate_only:
        count = len(set(args.app_id)) if args.app_id else len(source_apps)
        print(json.dumps({"status": "valid", "app_count": count}, ensure_ascii=False))
        return 0

    load_dotenv(args.env_file)
    result = extract_directory(
        args.input_dir,
        args.output,
        id_map_path=id_map,
        prompt_path=args.prompt,
        batch_size=args.batch_size,
        selected_app_ids=args.app_id,
        force=args.force,
        continue_on_error=args.continue_on_error,
        mock=args.mock,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["failed_batch_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
