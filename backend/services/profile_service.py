import json
from pathlib import Path


def load_theme_profile(theme_id, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    path = root / "data" / "styles" / theme_id / "theme.json"
    if not path.exists():
        return {"theme_id": theme_id, "description": "", "examples": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("theme_id", theme_id)
    data.setdefault("description", "")
    data.setdefault("examples", {})
    return data


def load_target_profile(target_app, root_dir=None):
    root = Path(root_dir) if root_dir else Path.cwd()
    path = root / "data" / "targets" / target_app / "target.json"
    if not path.exists():
        return {
            "app": target_app,
            "display_name": target_app,
            "category": "",
            "core_function": "",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("app", target_app)
    data.setdefault("display_name", target_app)
    data.setdefault("category", "")
    data.setdefault("core_function", "")
    return data
