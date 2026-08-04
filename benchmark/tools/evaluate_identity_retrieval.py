from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.services.dino_dense_service import dense_correspondence, extract_dense_features
from evaluation.services.embedding_service import TpqsConfig


def _load_pairs(dataset: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with (dataset / "pair_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["validation_status"] == "VALID":
                row["original_path"] = PROJECT_ROOT / row["original_asset_path"]
                row["themed_path"] = PROJECT_ROOT / row["themed_asset_path"]
                grouped[row["theme_id"]].append(row)
    return dict(grouped)


def _evaluate_theme(rows: list[dict], features: dict) -> dict:
    scores = np.empty((len(rows), len(rows)), dtype=np.float32)
    for query_index, query in enumerate(rows):
        themed = features[str(query["themed_path"])]
        for candidate_index, candidate in enumerate(rows):
            original = features[str(candidate["original_path"])]
            scores[query_index, candidate_index] = dense_correspondence(themed, original)["score"]

    ranks = []
    reciprocal_ranks = []
    aucs = []
    margins = []
    query_rows = []
    for index, row in enumerate(rows):
        matched = float(scores[index, index])
        distractors = np.delete(scores[index], index)
        rank = int(1 + np.sum(distractors > matched))
        auc = float(np.mean(matched > distractors) + 0.5 * np.mean(matched == distractors))
        hardest = float(np.max(distractors)) if len(distractors) else matched
        margin = matched - hardest
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        aucs.append(auc)
        margins.append(margin)
        query_rows.append(
            {
                "canonical_app_id": row["canonical_app_id"],
                "app_slug": row["app_slug"],
                "matched_score": matched,
                "hardest_distractor_score": hardest,
                "top1_margin": margin,
                "rank": rank,
                "candidate_count": len(rows),
                "auc": auc,
            }
        )
    query_rows.sort(key=lambda item: (item["rank"], -item["top1_margin"]), reverse=True)
    return {
        "pair_count": len(rows),
        "top1_accuracy": float(np.mean(np.asarray(ranks) == 1)),
        "top5_accuracy": float(np.mean(np.asarray(ranks) <= 5)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "pairwise_auc": float(np.mean(aucs)),
        "matched_score_mean": float(np.mean(np.diag(scores))),
        "distractor_score_mean": float(np.mean(scores[~np.eye(len(rows), dtype=bool)])) if len(rows) > 1 else 0.0,
        "top1_margin_mean": float(np.mean(margins)),
        "non_top1_count": int(np.sum(np.asarray(ranks) > 1)),
        "queries": query_rows,
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# DINOv3 dense identity retrieval on real pairs",
        "",
        "Each designer-themed icon is a query. Candidate identities are the original icons from the same theme. "
        "The correct candidate is defined only by the verified canonical label; other labels are objective distractors.",
        "",
        f"- Device: `{result['config']['device']}`",
        f"- Model: `{result['config']['model_id']}`",
        f"- Total verified pairs: {result['overall']['pair_count']}",
        "",
        "| Theme | pairs | top-1 | top-5 | MRR | pairwise AUC | mean top-1 margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for theme, item in result["themes"].items():
        lines.append(
            f"| {theme} | {item['pair_count']} | {item['top1_accuracy']:.1%} | {item['top5_accuracy']:.1%} "
            f"| {item['mean_reciprocal_rank']:.3f} | {item['pairwise_auc']:.3f} | {item['top1_margin_mean']:.4f} |"
        )
    overall = result["overall"]
    lines.extend(
        [
            f"| **macro/overall** | {overall['pair_count']} | {overall['top1_accuracy']:.1%} "
            f"| {overall['top5_accuracy']:.1%} | {overall['mean_reciprocal_rank']:.3f} "
            f"| {overall['pairwise_auc']:.3f} | {overall['top1_margin_mean']:.4f} |",
            "",
            "This test validates identity discrimination without generated degradations or human ratings. It does not "
            "by itself validate an absolute 0–100 calibration or a hard acceptance threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real-pair identity retrieval with cached DINOv3 dense features.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "benchmark" / "evaluation_set_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    grouped = _load_pairs(dataset)
    paths = list(
        dict.fromkeys(
            path
            for rows in grouped.values()
            for row in rows
            for path in (row["original_path"], row["themed_path"])
        )
    )
    config = TpqsConfig.from_env()
    print(f"Extracting/loading dense structure features for {len(paths)} unique images...")
    features = extract_dense_features(paths, config, root_dir=PROJECT_ROOT, view="structure")
    themes = {}
    all_queries = []
    for theme, rows in sorted(grouped.items()):
        print(f"Evaluating {theme}: {len(rows)} x {len(rows)} retrieval matrix...")
        themes[theme] = _evaluate_theme(rows, features)
        all_queries.extend(themes[theme]["queries"])

    pair_count = len(all_queries)
    overall = {
        "pair_count": pair_count,
        "top1_accuracy": float(np.mean([item["rank"] == 1 for item in all_queries])),
        "top5_accuracy": float(np.mean([item["rank"] <= 5 for item in all_queries])),
        "mean_reciprocal_rank": float(np.mean([1.0 / item["rank"] for item in all_queries])),
        "pairwise_auc": float(np.mean([item["auc"] for item in all_queries])),
        "top1_margin_mean": float(np.mean([item["top1_margin"] for item in all_queries])),
        "non_top1_count": int(sum(item["rank"] > 1 for item in all_queries)),
    }
    result = {
        "dataset_fingerprint": "a9982485db04a7f88e76236e227a1a65cb37f5b1a4f02057c3d919d49355e826",
        "protocol": "same-theme verified-label retrieval; full candidate gallery",
        "config": {
            "device": config.device,
            "model_id": config.model_id,
            "image_size": config.image_size,
            "view": "structure",
        },
        "themes": themes,
        "overall": overall,
    }
    output = args.output or dataset / "results" / "identity_retrieval_dinov3"
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "retrieval_report.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"themes": {key: {k: v for k, v in item.items() if k != "queries"} for key, item in themes.items()}, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
