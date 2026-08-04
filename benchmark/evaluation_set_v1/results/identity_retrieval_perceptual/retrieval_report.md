# Perceptual identity retrieval on real pairs

Each real themed icon is compared with its verified original and up to 10 deterministically sampled different-label originals from the same theme.

| Metric | top-1 | top-5 | MRR | sampled pairwise AUC |
|---|---:|---:|---:|---:|
| dists_structure | 18.4% | 50.0% | 0.349 | 0.543 |
| lpips_content | 17.1% | 51.3% | 0.348 | 0.559 |

This is an objective label-retrieval test using real assets and different-App negatives. It validates relative discrimination, not an absolute acceptance threshold.
