from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tpqs_workflow import run_tpqs


THEME_ID = "theme_001"
PACKAGE_ID = "package_001_theme_001"
EVAL_ID = "eval_001_package_001_theme_001"


def main() -> None:
    result = run_tpqs(THEME_ID, PACKAGE_ID, EVAL_ID)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
