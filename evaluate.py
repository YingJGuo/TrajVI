#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from data import _resize_frames, load_frames, load_manifest, load_masks, natural_key


METRICS = (
    "PSNR", "SSIM", "MSE", "PSNRCrop", "SSIMCrop", "SSIMCropFull", "MSECrop"
)
CSV_FIELDS = [
    "videoName", "Method", "PSNRavg", "PSNRstd", "SSIMavg", "SSIMstd",
    "MSEavg", "MSEstd", "PSNRCropavg", "PSNRCropstd", "SSIMCropavg",
    "SSIMCropstd", "SSIMCropFullavg", "SSIMCropFullstd", "MSECropavg",
    "MSECropstd", "valid_frames", "total_frames",
]


def _prediction_paths(folder: Path):
    paths = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(paths, key=lambda path: natural_key(path.name))


def _read_rgb(path: Path, size):
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _evaluate_video(output_root: Path, data_root: Path, dataset_name: str,
                    video_name: str, frame_dir: str, mask_dir: str,
                    storage_format: str, max_frames: int, dilation: int):
    pred_dir = output_root / video_name / "overlaid" / "notshifted" / "frameresult"
    pred_paths = _prediction_paths(pred_dir)
    frames = load_frames(
        data_root, dataset_name, video_name, frame_dir,
        storage_format, max_frames)
    frames, size = _resize_frames(frames, (288, 288))
    masks = load_masks(
        data_root, dataset_name, video_name, mask_dir, storage_format,
        len(frames), size, dilation)
    total = min(len(pred_paths), len(frames), len(masks), int(max_frames))
    if total <= 0:
        raise RuntimeError(f"No aligned frames for {video_name}")

    values = {name: [] for name in METRICS}
    valid = 0
    for index in range(total):
        gt = np.asarray(frames[index], dtype=np.uint8)
        pred = _read_rgb(pred_paths[index], size)
        mask = np.asarray(masks[index].convert("L")) > 0
        if int(mask.sum()) < 30:
            continue

        full_ssim, ssim_map = structural_similarity(
            gt, pred, channel_axis=2, data_range=255, full=True)
        crop_gt = gt[mask]
        crop_pred = pred[mask]
        values["PSNR"].append(peak_signal_noise_ratio(gt, pred, data_range=255))
        values["SSIM"].append(full_ssim)
        values["MSE"].append(mean_squared_error(gt, pred))
        values["PSNRCrop"].append(
            peak_signal_noise_ratio(crop_gt, crop_pred, data_range=255))
        values["SSIMCrop"].append(
            structural_similarity(crop_gt, crop_pred, channel_axis=1,
                                  data_range=255))
        values["SSIMCropFull"].append(
            float(np.mean(np.mean(ssim_map, axis=2)[mask])))
        values["MSECrop"].append(mean_squared_error(crop_gt, crop_pred))
        valid += 1

    if valid == 0:
        raise RuntimeError(f"No valid masked frames for {video_name}")
    row = {"videoName": video_name, "Method": ""}
    for name, sequence in values.items():
        row[f"{name}avg"] = float(np.mean(sequence))
        row[f"{name}std"] = float(np.std(sequence))
    row["valid_frames"] = valid
    row["total_frames"] = total
    return row


def _write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrajVI frame results")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-name", default="Endo_STTN")
    parser.add_argument("--split", default="test")
    parser.add_argument("--storage-format", choices=["zip", "folder"], default="zip")
    parser.add_argument("--frame-dir", default="JPEGImages")
    parser.add_argument("--mask-dir", default="AnnotationsShifted")
    parser.add_argument("--max-frames", type=int, default=927)
    parser.add_argument("--mask-dilation", type=int, default=8)
    parser.add_argument("--include-video", action="append", default=[])
    parser.add_argument("--exclude-video", action="append", default=[])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest = load_manifest(data_root, args.dataset_name, args.split)
    included = set(args.include_video)
    excluded = set(args.exclude_video)
    videos = [
        name for name in manifest
        if (not included or name in included) and name not in excluded
    ]
    if not videos:
        raise ValueError("No videos selected for evaluation")

    rows = []
    output_root = Path(args.output_root)
    for index, video_name in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video_name}", flush=True)
        row = _evaluate_video(
            output_root, data_root, args.dataset_name, video_name,
            args.frame_dir, args.mask_dir, args.storage_format,
            args.max_frames, args.mask_dilation)
        row["Method"] = args.method
        rows.append(row)

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    _write_csv(result_root / "quant_results.csv", rows, CSV_FIELDS)

    macro = {"method_name": args.method}
    for name in ("PSNRCrop", "SSIMCropFull", "MSECrop",
                 "PSNR", "SSIM", "MSE", "SSIMCrop"):
        macro[f"{name}avg"] = float(np.mean([row[f"{name}avg"] for row in rows]))
    _write_csv(result_root / "official_cpu_metrics_macro.csv", [macro],
               list(macro))

    total_valid = sum(float(row["valid_frames"]) for row in rows)
    weighted = {"method_name": args.method}
    for name in ("PSNRCrop", "SSIMCropFull", "MSECrop",
                 "PSNR", "SSIM", "MSE", "SSIMCrop"):
        weighted[f"{name}avg"] = float(sum(
            row[f"{name}avg"] * float(row["valid_frames"])
            for row in rows) / max(total_valid, 1.0))
    _write_csv(result_root / "official_cpu_metrics_weighted.csv", [weighted],
               list(weighted))

    protocol = {
        "metric": "skimage CPU metrics, data_range=255",
        "aggregation": "video macro average; weighted companion also saved",
        "data_root": str(data_root),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "storage_format": args.storage_format,
        "frame_dir": args.frame_dir,
        "mask_dir": args.mask_dir,
        "mask_dilation": args.mask_dilation,
        "max_frames": args.max_frames,
        "videos": videos,
    }
    (result_root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMacro average:")
    print("  PSNR-Crop       %.6f" % macro["PSNRCropavg"])
    print("  SSIM-Crop-Full  %.6f" % macro["SSIMCropFullavg"])
    print("  MSE-Crop        %.6f" % macro["MSECropavg"])
    print(f"Saved metrics to {result_root}")


if __name__ == "__main__":
    main()
