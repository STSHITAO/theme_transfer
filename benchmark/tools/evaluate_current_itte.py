from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.services.dino_dense_service import extract_dense_features
from evaluation.services.embedding_service import TpqsConfig
from evaluation.services.eval_path_service import GeneratedIcon, ResolvedEvalInputs, ThemeTransferExample
from evaluation.services.itte_v12_service import compute_itte_v12_metrics
from evaluation.services.style_feature_service import extract_style_feature_groups, extract_style_features


DATASET_ROOT = PROJECT_ROOT / "benchmark" / "evaluation_set_v1"
PAIR_MANIFEST = DATASET_ROOT / "pair_manifest.csv"
DEFAULT_OUTPUT = DATASET_ROOT / "results" / "itte_current_evaluation"
THEME_IDS = ("theme_001", "theme_002", "theme_003", "theme_004")
SCENARIOS = ("designer_positive", "original_control")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def project_path(value: str) -> Path:
    return (PROJECT_ROOT / value).resolve()


def stable_seed(seed: int, theme_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{theme_id}".encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")


def select_split(rows: list[dict[str, str]], theme_id: str, references: int, queries: int, seed: int) -> dict:
    themed = [row for row in rows if row["theme_id"] == theme_id and row["validation_status"] == "VALID"]
    shuffled = sorted(themed, key=lambda row: row["canonical_app_id"])
    random.Random(stable_seed(seed, theme_id)).shuffle(shuffled)
    required = references + queries
    if len(shuffled) < required:
        raise ValueError(f"{theme_id} has {len(shuffled)} pairs, but {required} are required")
    reference_rows = shuffled[:references]
    query_rows = shuffled[references:required]
    reference_ids = {row["canonical_app_id"] for row in reference_rows}
    query_ids = {row["canonical_app_id"] for row in query_rows}
    if reference_ids & query_ids:
        raise AssertionError(f"Reference/query identity leakage in {theme_id}")
    return {"theme_id": theme_id, "reference_rows": reference_rows, "query_rows": query_rows}


def resolved_inputs(split: dict, scenario: str) -> ResolvedEvalInputs:
    theme_examples = [
        ThemeTransferExample(
            app=row["app_slug"],
            original_path=project_path(row["original_asset_path"]),
            style_ref_path=project_path(row["themed_asset_path"]),
            reference_raw_path=project_path(row["original_asset_path"]),
        )
        for row in split["reference_rows"]
    ]
    generated_icons = []
    target_originals = {}
    for row in split["query_rows"]:
        original = project_path(row["original_asset_path"])
        themed = project_path(row["themed_asset_path"])
        generated_icons.append(
            GeneratedIcon(
                app=row["app_slug"],
                path=themed if scenario == "designer_positive" else original,
            )
        )
        target_originals[row["app_slug"]] = original
    return ResolvedEvalInputs(
        theme_id=split["theme_id"],
        package_id=f"benchmark_{split['theme_id']}_{scenario}",
        theme_examples=theme_examples,
        theme_refs=[item.style_ref_path for item in theme_examples],
        generated_icons=generated_icons,
        target_originals=target_originals,
        missing_apps=[],
        skipped_apps=[],
    )


def all_paths(resolved_runs: list[ResolvedEvalInputs]) -> list[Path]:
    paths = []
    for resolved in resolved_runs:
        paths.extend(resolved.theme_refs)
        paths.extend(item.reference_raw_path for item in resolved.theme_examples)
        paths.extend(item.path for item in resolved.generated_icons)
        paths.extend(resolved.target_originals[item.app] for item in resolved.generated_icons)
    return list(dict.fromkeys(paths))


def component_summary(report: dict) -> dict:
    style = report["style_fidelity"]
    identity = report["identity_preservation"]
    return {
        "itte_score": report["itte_score"],
        "style_fidelity_score": report["style_fidelity_score"],
        "identity_preservation_score": report["identity_preservation_score"],
        "package_coherence_score": report["package_coherence_score"],
        "visual_quality_score": report["visual_quality_score"],
        "identity_p10": identity["p10_score"],
        "decision": report["decision"],
        "confidence": report["evaluation_confidence"],
        "hard_failure_count": len(report["hard_failures"]),
        "hard_failures": report["hard_failures"],
        "style_available_weight": style["available_weight"],
        "identity_available_weight": identity["available_weight"],
        "style_components": {
            name: {
                "score": item.get("score"),
                "reliable": item.get("reliable"),
                "weight": item.get("weight"),
            }
            for name, item in style["components"].items()
        },
        "identity_components": {
            name: {
                "score": item.get("score"),
                "reliable": item.get("reliable"),
                "weight": item.get("weight"),
            }
            for name, item in identity["components"].items()
        },
        "quality_hard_failure_count": len(report["visual_quality"]["hard_failures"]),
        "package_outlier_apps": report["package_coherence"]["outlier_apps"],
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(run_rows: list[dict]) -> dict:
    fields = (
        "itte_score",
        "style_fidelity_score",
        "identity_preservation_score",
        "package_coherence_score",
        "visual_quality_score",
        "identity_p10",
    )
    by_scenario = {}
    for scenario in SCENARIOS:
        selected = [row for row in run_rows if row["scenario"] == scenario]
        if not selected:
            continue
        by_scenario[scenario] = {
            "run_count": len(selected),
            "macro_means": {field: mean([float(row[field]) for row in selected]) for field in fields},
            "by_theme": {row["theme_id"]: {field: row[field] for field in fields} for row in selected},
            "hard_failure_count": sum(int(row["hard_failure_count"]) for row in selected),
        }
    comparison = {}
    if all(scenario in by_scenario for scenario in SCENARIOS):
        for field in fields:
            positive = by_scenario["designer_positive"]["macro_means"][field]
            control = by_scenario["original_control"]["macro_means"][field]
            comparison[field] = {"designer_positive": positive, "original_control": control, "delta": positive - control}
    return {"by_scenario": by_scenario, "designer_minus_original_control": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the current ITTE implementation on the real designer benchmark.")
    parser.add_argument("--themes", nargs="+", choices=THEME_IDS, default=list(THEME_IDS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--references", type=int, default=8)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    rows = read_csv(PAIR_MANIFEST)
    splits = [select_split(rows, theme_id, args.references, args.queries, args.seed) for theme_id in args.themes]
    planned = [(split, scenario, resolved_inputs(split, scenario)) for split in splits for scenario in args.scenarios]
    resolved_runs = [item[2] for item in planned]
    paths = all_paths(resolved_runs)

    config = TpqsConfig.from_env()
    if config.embedding_backend != "dinov3":
        raise RuntimeError("Official Benchmark evaluation requires the dinov3 backend")

    print(f"Precomputing shared features for {len(paths)} unique images...", flush=True)
    style_features = extract_style_features(paths, config, root_dir=PROJECT_ROOT)
    style_groups = extract_style_feature_groups(paths, config, root_dir=PROJECT_ROOT)
    dense_structure = extract_dense_features(paths, config, root_dir=PROJECT_ROOT, view="structure")
    dense_appearance = extract_dense_features(paths, config, root_dir=PROJECT_ROOT, view="appearance")

    run_rows = []
    run_index = []
    for split, scenario, resolved in planned:
        run_id = f"{split['theme_id']}__{scenario}"
        print(f"Running {run_id}...", flush=True)
        metrics = compute_itte_v12_metrics(
            resolved,
            style_features,
            style_groups,
            dense_structure,
            dense_appearance,
            config,
            PROJECT_ROOT,
        )
        report = metrics.report
        summary = component_summary(report)
        row = {"run_id": run_id, "theme_id": split["theme_id"], "scenario": scenario, **summary}
        run_rows.append(row)
        split_payload = {
            "theme_id": split["theme_id"],
            "scenario": scenario,
            "reference_apps": [item["app_slug"] for item in split["reference_rows"]],
            "query_apps": [item["app_slug"] for item in split["query_rows"]],
            "identity_disjoint": True,
        }
        write_json(output_root / "runs" / run_id / "split.json", split_payload)
        write_json(output_root / "runs" / run_id / "itte_report.json", report)
        write_json(output_root / "runs" / run_id / "style_pairwise_distances.json", metrics.style_pairwise)
        write_json(output_root / "runs" / run_id / "style_delta_distances.json", metrics.style_delta)
        write_json(output_root / "runs" / run_id / "dino_pairwise_distances.json", metrics.dino_pairwise)
        run_index.append({"run_id": run_id, "split": split_payload, "summary": summary})
        print(json.dumps({"run_id": run_id, **summary}, ensure_ascii=False), flush=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": f"Evaluate {report['itte_version'] if run_index else 'current ITTE'} on real designer positives and the natural no-transfer original control.",
        "dataset_fingerprint": "a9982485db04a7f88e76236e227a1a65cb37f5b1a4f02057c3d919d49355e826",
        "config": asdict(config),
        "sampling": {
            "themes": args.themes,
            "scenarios": args.scenarios,
            "references_per_theme": args.references,
            "queries_per_theme": args.queries,
            "seed": args.seed,
            "reference_query_identity_disjoint": True,
        },
        "runs": run_index,
        "aggregate": aggregate(run_rows),
    }
    write_json(output_root / "baseline_summary.json", payload)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
