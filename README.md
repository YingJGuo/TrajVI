# TrajVI

Inference-only release of the Full V4 model for the EndoSTTN data format.

The released pipeline is:

```text
completed RAFT flow
  -> full-video coarse repair
  -> 60-frame repaired-context CoTracker3 trajectories
  -> TLP (long-term region-warp DCN)
  -> TTR (sparse feature trajectory transformer)
  -> SparseTransformer + decoder
```

This directory intentionally excludes training code, CoWTracker, ablation
routes, profiling tools, and experiment-specific visualizers.

## Dependencies

Install PyTorch and torchvision for the CUDA version on the target machine,
then install the small Python dependency set:

```bash
pip install -r requirements.txt
```

CoTracker3 is kept as an external dependency because its repository and
checkpoint are large. Clone the compatible CoTracker repository separately
and pass its path with `--cotracker-repo`.

The inference command also needs four weights. They are intentionally not
committed to this source tree because of their size and their separate
licenses; see [weights/README.md](weights/README.md) for the expected
filenames and locations.

- the released Full V4 generator checkpoint;
- the official RAFT checkpoint;
- the flow-completion checkpoint;
- the CoTracker3 Offline checkpoint.

## Dataset Layout

The historical EndoSTTN zip layout is supported directly:

```text
DATA_ROOT/
  DATASET_NAME/
    train.json
    test.json
    JPEGImages/
      video_name.zip
    AnnotationsShifted/
      video_name.zip
```

Each zip contains one image per frame. Folder-backed videos are also
supported with `--storage-format folder`; in that mode each video is a
subdirectory under both `--frame-dir` and `--mask-dir`.

## Inference

The following is the production Quality protocol used by Full V4:

- 288 x 288 processing resolution;
- 10-frame local windows with stride 5;
- 60-frame, temporal-stride-1 trajectory context;
- 512 support queries interpolated to at most 2048 masked feature queries;
- CoTracker3 update iterations = 2;
- mask-internal nearest-neighbor support interpolation with `k=4`;
- fixed TLP update gate = 0.25;
- frame results saved under the official `overlaid/notshifted/frameresult`
  layout.

Example:

```bash
python infer.py \
  --data-root /path/to/data \
  --dataset-name Endo_STTN \
  --split test \
  --storage-format zip \
  --frame-dir JPEGImages \
  --mask-dir AnnotationsShifted \
  --checkpoint /path/to/full_v4/gen_015000.pth \
  --raft-checkpoint /path/to/raft-things.pth \
  --flow-checkpoint /path/to/flow_completion.pth \
  --cotracker-checkpoint /path/to/scaled_offline.pth \
  --cotracker-repo /path/to/co-tracker \
  --output-root /path/to/results/full_v4 \
  --gpus 0 \
  --skip-existing
```

PNG is the default frame-result format so that evaluation does not include
JPEG artifacts. Use `--frame-format jpg --frame-quality 95` when smaller
output is preferred.

Use `--video VIDEO_NAME` one or more times to run a subset. Use more than one
GPU by passing several IDs, for example `--gpus 0 1 2 3`.

## Evaluation

Evaluate the saved frame results with CPU/skimage metrics:

```bash
python evaluate.py \
  --output-root /path/to/results/full_v4 \
  --result-root /path/to/results/full_v4_metrics \
  --method TrajVI-Full-V4 \
  --data-root /path/to/data \
  --dataset-name Endo_STTN \
  --split test \
  --storage-format zip \
  --frame-dir JPEGImages \
  --mask-dir AnnotationsShifted \
  --mask-dilation 8
```

The evaluator writes per-video results, macro-average metrics, weighted
metrics, and the exact protocol to `result-root`.

The principal table columns are `PSNRCropavg`, `SSIMCropFullavg`, and
`MSECropavg`. `SSIMCropFullavg` is the mean of the full-image SSIM map inside
the mask, matching the EndoSTTN/DAEVI CPU evaluation protocol.
