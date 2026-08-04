from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.services.dino_dense_service import dense_correspondence, extract_dense_features
from evaluation.services.embedding_service import TpqsConfig
from evaluation.services.perceptual_service import compute_perceptual_scores
from evaluation.services.vgg_gram_service import compute_vgg_gram_style_fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise ITTE model families through one CPU or CUDA device interface.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = PROJECT_ROOT / "benchmark" / "evaluation_set_v1"
    with (dataset / "pair_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["validation_status"] == "VALID"][:3]
    originals = [PROJECT_ROOT / row["original_asset_path"] for row in rows]
    themed = [PROJECT_ROOT / row["themed_asset_path"] for row in rows]
    config = replace(
        TpqsConfig.from_env(),
        device=args.device,
        image_size=args.image_size,
        batch_size=1,
    )

    dense = extract_dense_features([originals[0], themed[0]], config, PROJECT_ROOT, view="structure")
    dino = dense_correspondence(dense[str(originals[0])], dense[str(themed[0])])
    vgg = compute_vgg_gram_style_fit(
        themed[:2], themed[2:], originals[2:], PROJECT_ROOT, True,
        device=args.device, image_size=args.image_size,
    )
    perceptual = compute_perceptual_scores(
        originals[:2], themed[:2], originals[2:], themed[2:], [rows[2]["app_slug"]],
        PROJECT_ROOT, args.device, True, image_size=args.image_size, batch_size=1,
    )
    result = {
        "requested_device": args.device,
        "image_size": args.image_size,
        "dino_dense": {"ok": True, **dino},
        "vgg_gram": {
            "ok": vgg.get("score") is not None,
            "device": vgg.get("device"),
            "cache": vgg.get("cache"),
        },
        "perceptual": {
            "ok": perceptual["available"],
            "device": perceptual["device"],
            "models": perceptual["models"],
            "cache": perceptual.get("cache"),
        },
    }
    output = args.output or dataset / "results" / f"device_smoke_{args.device.replace(':', '_')}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all((result["dino_dense"]["ok"], result["vgg_gram"]["ok"], result["perceptual"]["ok"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
