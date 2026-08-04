from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.services.embedding_service import TpqsConfig
from evaluation.services.perceptual_service import (
    _compute_dists_pair_components,
    _compute_lpips_pairs,
    _distance,
    _load_tensor,
)


def _load_pairs(dataset: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with (dataset / "pair_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["validation_status"] == "VALID":
                row["original_path"] = PROJECT_ROOT / row["original_asset_path"]
                row["themed_path"] = PROJECT_ROOT / row["themed_asset_path"]
                grouped[row["theme_id"]].append(row)
    return dict(grouped)


def _sample_candidates(grouped: dict[str, list[dict]], distractors: int, seed: int):
    queries = []
    pairs = []
    for theme, rows in sorted(grouped.items()):
        for index, row in enumerate(rows):
            candidates = [candidate for candidate_index, candidate in enumerate(rows) if candidate_index != index]
            stable = int.from_bytes(
                hashlib.sha256(f"{seed}:{theme}:{row['canonical_app_id']}".encode("utf-8")).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(stable)
            count = min(distractors, len(candidates))
            selected = [candidates[item] for item in rng.choice(len(candidates), size=count, replace=False)]
            candidate_paths = [row["original_path"], *[item["original_path"] for item in selected]]
            query_pairs = [(row["themed_path"], candidate) for candidate in candidate_paths]
            pairs.extend(query_pairs)
            queries.append(
                {
                    "theme": theme,
                    "app_slug": row["app_slug"],
                    "canonical_app_id": row["canonical_app_id"],
                    "pairs": query_pairs,
                }
            )
    return queries, list(dict.fromkeys(pairs))


def _evaluate(queries: list[dict], distances: dict, component: str) -> dict:
    rows = []
    for query in queries:
        values = [_distance(distances, left, right, component) for left, right in query["pairs"]]
        matched = values[0]
        distractors = np.asarray(values[1:], dtype=np.float64)
        rank = int(1 + np.sum(distractors < matched))
        auc = float(np.mean(matched < distractors) + 0.5 * np.mean(matched == distractors))
        rows.append(
            {
                "theme": query["theme"],
                "canonical_app_id": query["canonical_app_id"],
                "app_slug": query["app_slug"],
                "matched_distance": matched,
                "hardest_distractor_distance": float(np.min(distractors)),
                "rank": rank,
                "candidate_count": len(values),
                "auc": auc,
            }
        )
    by_theme = {}
    for theme in sorted({row["theme"] for row in rows}):
        selected = [row for row in rows if row["theme"] == theme]
        by_theme[theme] = _summarize(selected)
    return {"overall": _summarize(rows), "by_theme": by_theme, "queries": rows}


def _summarize(rows: list[dict]) -> dict:
    return {
        "query_count": len(rows),
        "top1_accuracy": float(np.mean([row["rank"] == 1 for row in rows])),
        "top5_accuracy": float(np.mean([row["rank"] <= 5 for row in rows])),
        "mean_reciprocal_rank": float(np.mean([1.0 / row["rank"] for row in rows])),
        "sampled_pairwise_auc": float(np.mean([row["auc"] for row in rows])),
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Perceptual identity retrieval on real pairs",
        "",
        f"Each real themed icon is compared with its verified original and up to {result['protocol']['distractors_per_query']} "
        "deterministically sampled different-label originals from the same theme.",
        "",
        "| Metric | top-1 | top-5 | MRR | sampled pairwise AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("dists_structure", "lpips_content"):
        item = result[name]["overall"]
        lines.append(
            f"| {name} | {item['top1_accuracy']:.1%} | {item['top5_accuracy']:.1%} "
            f"| {item['mean_reciprocal_rank']:.3f} | {item['sampled_pairwise_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This is an objective label-retrieval test using real assets and different-App negatives. It validates "
            "relative discrimination, not an absolute acceptance threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DISTS/LPIPS identity retrieval on verified real pairs.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "benchmark" / "evaluation_set_v1")
    parser.add_argument("--distractors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = TpqsConfig.from_env()
    grouped = _load_pairs(args.dataset.resolve())
    queries, pairs = _sample_candidates(grouped, max(args.distractors, 1), args.seed)
    unique_paths = list(dict.fromkeys(path for pair in pairs for path in pair))

    import torch
    import lpips
    from DISTS_pytorch import DISTS

    actual_device = config.device if not config.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    torch_home = PROJECT_ROOT / "models" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    cache_dir = PROJECT_ROOT / "data" / "evaluations" / "_cache" / "perceptual_pairs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {len(unique_paths)} structure views for {len(pairs)} unique sampled pairs...")
    tensors = {str(path): _load_tensor(path, config.image_size, torch, "structure") for path in unique_paths}

    print("Computing/loading DISTS structure distances...")
    dists_model = DISTS(load_weights=True).to(actual_device).eval()
    dists_distances = _compute_dists_pair_components(
        dists_model,
        pairs,
        tensors,
        actual_device,
        torch,
        batch_size=max(config.batch_size, 1),
        cache_dir=cache_dir,
        image_size=config.image_size,
        view="structure",
    )
    del dists_model
    if actual_device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("Computing/loading LPIPS content distances...")
    lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(actual_device).eval()
    lpips_distances = _compute_lpips_pairs(
        lpips_model,
        pairs,
        tensors,
        actual_device,
        torch,
        batch_size=max(config.batch_size, 1),
        cache_dir=cache_dir,
        image_size=config.image_size,
        view="structure",
    )

    result = {
        "protocol": {
            "verified_real_pairs": sum(len(rows) for rows in grouped.values()),
            "distractors_per_query": args.distractors,
            "seed": args.seed,
            "same_theme_distractors": True,
        },
        "config": {"device": actual_device, "image_size": config.image_size},
        "dists_structure": _evaluate(queries, dists_distances, "structure"),
        "lpips_content": _evaluate(queries, lpips_distances, "distance"),
    }
    output = args.output or args.dataset.resolve() / "results" / "identity_retrieval_perceptual"
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "retrieval_report.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({name: result[name]["overall"] for name in ("dists_structure", "lpips_content")}, indent=2))


if __name__ == "__main__":
    main()
