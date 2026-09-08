# Bundled CoTracker3 source

This directory contains the CoTracker3 inference source used by TrajVI. It is
derived from [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)
and distributed under the included license.

TrajVI uses the offline predictor with two small compatibility extensions:

- `return_scores=True` returns visibility and confidence maps;
- `iters` selects the number of tracker update iterations.

The bundled source is loaded automatically by `infer.py`. The `--cotracker-repo`
argument is available only when replacing it with another compatible checkout.
