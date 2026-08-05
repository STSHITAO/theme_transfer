"""Create a reproducibility manifest for generated experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOTS = (
    Path("data/packages"),
    Path("data/evaluations"),
    Path("benchmark/evaluation_set_v1/results"),
)
EXCLUDED_PARTS = {"_cache", "__pycache__", ".final_staging_", ".final_backup_"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_included(relative_path: Path) -> bool:
    return not any(
        part in EXCLUDED_PARTS or part.startswith(".final_staging_") or part.startswith(".final_backup_")
        for part in relative_path.parts
    )


def build_manifest(root: Path) -> dict:
    root = root.resolve()
    artifacts = []
    summary = {}
    for include_root in INCLUDE_ROOTS:
        source = root / include_root
        key = include_root.as_posix()
        files = []
        if source.exists():
            for path in sorted(item for item in source.rglob("*") if item.is_file()):
                relative_path = path.relative_to(root)
                if not is_included(relative_path):
                    continue
                record = {
                    "path": relative_path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                artifacts.append(record)
                files.append(record)
        summary[key] = {
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files),
        }
    return {
        "schema_version": "experiment-artifacts-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "included_roots": [item.as_posix() for item in INCLUDE_ROOTS],
        "excluded_path_parts": sorted(EXCLUDED_PARTS),
        "summary": summary,
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/EXPERIMENT_MANIFEST.json"),
        help="Manifest path, relative to --root unless absolute.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": manifest["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
