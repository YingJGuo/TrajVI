#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from data import (
    _resize_frames,
    get_ref_ids,
    image_list_to_tensor,
    load_frames,
    load_manifest,
    load_masks,
    local_ids,
    mask_list_to_tensor,
)


def _completed_flows(frames, masks, raft, flow_completion, raft_iters,
                     subvideo_length):
    video_length = int(frames.shape[1])
    if video_length < 2:
        raise ValueError("At least two frames are required for RAFT")

    short_clip_len = 12 if frames.shape[-1] <= 640 else 8
    if frames.shape[-1] > 720:
        short_clip_len = 4
    if frames.shape[-1] > 1280:
        short_clip_len = 2

    with torch.no_grad():
        forward_parts, backward_parts = [], []
        for start in range(0, video_length, short_clip_len):
            end = min(video_length, start + short_clip_len)
            clip = frames[:, start:end] if start == 0 else frames[:, start - 1:end]
            flow_f, flow_b = raft(clip, iters=raft_iters)
            forward_parts.append(flow_f)
            backward_parts.append(flow_b)
        raw_flows = (torch.cat(forward_parts, dim=1),
                     torch.cat(backward_parts, dim=1))

        flow_length = int(raw_flows[0].shape[1])
        if flow_length <= int(subvideo_length):
            completed, _ = flow_completion.forward_bidirect_flow(
                raw_flows, masks)
            return flow_completion.combine_flow(raw_flows, completed, masks)

        forward_parts, backward_parts = [], []
        pad = 5
        for start in range(0, flow_length, int(subvideo_length)):
            source_start = max(0, start - pad)
            source_end = min(video_length, start + int(subvideo_length) + pad)
            trim_start = start - source_start
            trim_end = source_end - min(video_length, start + int(subvideo_length))
            raw_sub = (
                raw_flows[0][:, source_start:source_end],
                raw_flows[1][:, source_start:source_end],
            )
            completed_sub, _ = flow_completion.forward_bidirect_flow(
                raw_sub, masks[:, source_start:source_end + 1])
            completed_sub = flow_completion.combine_flow(
                raw_sub, completed_sub, masks[:, source_start:source_end + 1])
            forward_parts.append(
                completed_sub[0][:, trim_start:completed_sub[0].shape[1] - trim_end])
            backward_parts.append(
                completed_sub[1][:, trim_start:completed_sub[1].shape[1] - trim_end])
        return torch.cat(forward_parts, dim=1), torch.cat(backward_parts, dim=1)


def _model_window(model, frames, flows, masks, updated_masks, local_length,
                  trajectory_result=None, trajectory_context_frames=None,
                  trajectory_context_masks=None,
                  trajectory_context_local_indices=None,
                  trajectory_context_ids=None):
    return model(
        frames,
        flows,
        masks,
        updated_masks,
        local_length,
        trajectory_result=trajectory_result,
        trajectory_context_frames=trajectory_context_frames,
        trajectory_context_masks=trajectory_context_masks,
        trajectory_context_local_indices=trajectory_context_local_indices,
    )


def _composite_prediction(prediction, original_frames, mask_tensor, local_ids_):
    prediction = ((prediction.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    prediction = prediction[0].permute(0, 2, 3, 1).cpu().numpy() * 255.0
    masks = mask_tensor[0, local_ids_].float().cpu().permute(0, 2, 3, 1).numpy()
    result = []
    for index, frame_id in enumerate(local_ids_):
        frame = prediction[index] * masks[index]
        frame += original_frames[frame_id] * (1.0 - masks[index])
        result.append(np.clip(frame, 0, 255).astype(np.uint8))
    return result


def _save_video(save_root: Path, frames, fps, frame_format, frame_quality):
    save_root.mkdir(parents=True, exist_ok=True)
    output_path = save_root / "inpaint_out.mp4"
    imageio.mimwrite(output_path, frames, fps=fps, quality=7)

    frame_dir = save_root / "overlaid" / "notshifted" / "frameresult"
    frame_dir.mkdir(parents=True, exist_ok=True)
    extension = "png" if frame_format.lower() == "png" else "jpg"
    for index, frame in enumerate(frames):
        path = frame_dir / f"{index:05d}.{extension}"
        if extension == "png":
            imageio.imwrite(path, frame)
        else:
            imageio.imwrite(path, frame, quality=int(frame_quality),
                            subsampling=0)
    return output_path, frame_dir


def _load_models(args, device):
    from core.cotracker3_feature_trajectory import (
        CoTracker3FeatureTrajectoryExtractor,
    )
    from model.modules.flow_comp_raft import RAFT_bi
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet

    raft = RAFT_bi(args.raft_checkpoint, device=device)
    flow_completion = RecurrentFlowCompleteNet(args.flow_checkpoint)
    flow_completion = flow_completion.to(device).eval()
    for parameter in flow_completion.parameters():
        parameter.requires_grad = False

    model = InpaintGenerator(model_path=args.checkpoint).to(device).eval()

    trajectory = CoTracker3FeatureTrajectoryExtractor(
        checkpoint_path=args.cotracker_checkpoint,
        repo_path=args.cotracker_repo,
        device=str(device),
        context_window=60,
        query_batch_size=512,
        max_queries=2048,
        max_support_queries=512,
        support_interpolation_mode="nearest",
        support_interpolation_k=4,
        iterations=2,
    )
    return raft, flow_completion, model, trajectory


def _run_video(video_name, declared_frames, args, device, models):
    raft, flow_completion, model, trajectory_extractor = models
    output_root = Path(args.output_root) / video_name
    output_path = output_root / "inpaint_out.mp4"
    if args.skip_existing and output_path.exists():
        print(f"[skip] {video_name}")
        return

    print(f"[start] {video_name}", flush=True)
    frames_pil = load_frames(
        Path(args.data_root), args.dataset_name, video_name,
        args.frame_dir, args.storage_format, args.max_frames)
    frames_pil, process_size = _resize_frames(
        frames_pil, (args.width, args.height))
    frame_count = len(frames_pil)
    mask_pil = load_masks(
        Path(args.data_root), args.dataset_name, video_name,
        args.mask_dir, args.storage_format, frame_count,
        process_size, args.mask_dilation)

    original_frames = [np.asarray(frame, dtype=np.uint8) for frame in frames_pil]
    frames = image_list_to_tensor(frames_pil, device) * 2.0 - 1.0
    masks = mask_list_to_tensor(mask_pil, device)
    flow_masks = masks
    updated_frames = frames * (1.0 - masks)
    updated_masks = masks.clone()

    started = time.perf_counter()
    with torch.no_grad():
        completed_flows = _completed_flows(
            frames, flow_masks, raft, flow_completion,
            args.raft_iters, args.subvideo_length)

        local_stride = args.neighbor_length // 2
        reference_count = (
            args.subvideo_length // args.ref_stride
            if frame_count > args.subvideo_length else -1
        )

        coarse_frames = [None] * frame_count
        print(f"[coarse] {video_name}: {frame_count} frames", flush=True)
        for center in range(0, frame_count, local_stride):
            local = local_ids(center, frame_count, args.neighbor_length)
            refs = get_ref_ids(
                center, local, frame_count, args.ref_stride, reference_count)
            model_ids = local + refs
            local_flows = (
                completed_flows[0][:, local[:-1]],
                completed_flows[1][:, local[:-1]],
            )
            coarse = _model_window(
                model, updated_frames[:, model_ids], local_flows,
                masks[:, model_ids], updated_masks[:, model_ids], len(local))
            predictions = _composite_prediction(coarse, original_frames, masks, local)
            for frame_id, prediction in zip(local, predictions):
                if coarse_frames[frame_id] is None:
                    coarse_frames[frame_id] = prediction
                else:
                    coarse_frames[frame_id] = np.rint(
                        0.5 * coarse_frames[frame_id].astype(np.float32)
                        + 0.5 * prediction.astype(np.float32)
                    ).astype(np.uint8)

        coarse_frames = [
            original_frames[index] if frame is None else frame
            for index, frame in enumerate(coarse_frames)
        ]

        result_frames = [None] * frame_count
        print(f"[full] {video_name}: trajectory + TLP + TTR", flush=True)
        for center in range(0, frame_count, local_stride):
            local = local_ids(center, frame_count, args.neighbor_length)
            refs = get_ref_ids(
                center, local, frame_count, args.ref_stride, reference_count)
            model_ids = local + refs

            from core.cotracker3_feature_trajectory import build_long_term_context_ids

            context_ids = build_long_term_context_ids(
                center_frame=center,
                video_length=frame_count,
                local_frame_ids=local,
                window_size=60,
                temporal_stride=1,
            )
            local_context_indices = torch.tensor(
                [context_ids.index(frame_id) for frame_id in local],
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            context_masks = masks[:, context_ids]
            context_frames = updated_frames[:, context_ids]
            repaired_context = image_list_to_tensor(
                [Image.fromarray(coarse_frames[index]) for index in context_ids],
                device,
            ) * 2.0 - 1.0
            tracking_context = (
                frames[:, context_ids] * (1.0 - context_masks)
                + repaired_context * context_masks
            )
            trajectory_result = trajectory_extractor.compute(
                clean_context_frames=tracking_context,
                context_masks=context_masks,
                local_context_indices=local_context_indices,
                feature_size=(process_size[1] // 4, process_size[0] // 4),
            )
            local_flows = (
                completed_flows[0][:, local[:-1]],
                completed_flows[1][:, local[:-1]],
            )
            refined = _model_window(
                model, updated_frames[:, model_ids], local_flows,
                masks[:, model_ids], updated_masks[:, model_ids], len(local),
                trajectory_result=trajectory_result,
                trajectory_context_frames=context_frames,
                trajectory_context_masks=context_masks,
                trajectory_context_local_indices=local_context_indices,
                trajectory_context_ids=context_ids,
            )
            predictions = _composite_prediction(refined, original_frames, masks, local)
            for frame_id, prediction in zip(local, predictions):
                if result_frames[frame_id] is None:
                    result_frames[frame_id] = prediction
                else:
                    result_frames[frame_id] = np.rint(
                        0.5 * result_frames[frame_id].astype(np.float32)
                        + 0.5 * prediction.astype(np.float32)
                    ).astype(np.uint8)
            del trajectory_result, tracking_context, repaired_context

    result_frames = [
        original_frames[index] if frame is None else frame
        for index, frame in enumerate(result_frames)
    ]
    output_path, frame_dir = _save_video(
        output_root, result_frames, args.fps,
        args.frame_format, args.frame_quality)
    elapsed = time.perf_counter() - started
    print(
        f"[done] {video_name}: {output_path} | "
        f"{frame_count / max(elapsed, 1e-6):.3f} FPS | frames={frame_dir}",
        flush=True,
    )


def _worker(worker_id, gpu_id, video_items, args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"[worker {worker_id}] device={device}, videos={len(video_items)}")
    models = _load_models(args, device)
    for video_name, declared_frames in video_items:
        try:
            _run_video(video_name, declared_frames, args, device, models)
        except Exception:
            print(f"[failed] {video_name}", flush=True)
            import traceback
            traceback.print_exc()


def _split(items, count):
    buckets = [[] for _ in range(count)]
    loads = [0] * count
    for item in sorted(items, key=lambda value: value[1] or 10**9, reverse=True):
        index = loads.index(min(loads))
        buckets[index].append(item)
        loads[index] += int(item[1] or 1000)
    return buckets


def parse_args():
    parser = argparse.ArgumentParser(description="TrajVI Full V4 inference")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-name", default="Endo_STTN")
    parser.add_argument("--split", default="test")
    parser.add_argument("--storage-format", choices=["zip", "folder"], default="zip")
    parser.add_argument("--frame-dir", default="JPEGImages")
    parser.add_argument("--mask-dir", default="AnnotationsShifted")
    parser.add_argument("--checkpoint", required=True,
                        help="Full V4 generator checkpoint, e.g. gen_best_psnr.pth")
    parser.add_argument("--raft-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--cotracker-checkpoint", required=True)
    parser.add_argument(
        "--cotracker-repo", default=None,
        help="Optional CoTracker3 source directory; bundled source is used by default.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--max-frames", type=int, default=927)
    parser.add_argument("--mask-dilation", type=int, default=8)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--subvideo-length", type=int, default=80)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--frame-format", choices=["jpg", "png"], default="png",
        help="Saved frame-result format; PNG is the lossless default.",
    )
    parser.add_argument("--frame-quality", type=int, default=95)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--video", action="append", default=[],
                        help="Process only this video; may be repeated")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    manifest = load_manifest(data_root, args.dataset_name, args.split)
    items = list(manifest.items())
    if args.video:
        selected = set(args.video)
        items = [item for item in items if item[0] in selected]
    if not items:
        raise ValueError("No videos selected")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    buckets = _split(items, len(args.gpus))
    print(f"Loaded {len(items)} videos from {args.dataset_name}/{args.split}.json")
    print("Full V4 protocol: context=60, stride=1, queries=512->2048, "
          "iterations=2")

    if len(args.gpus) == 1:
        _worker(0, args.gpus[0], buckets[0], args)
        return
    processes = []
    for worker_id, (gpu_id, bucket) in enumerate(zip(args.gpus, buckets)):
        process = mp.Process(
            target=_worker, args=(worker_id, gpu_id, bucket, args))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode]
    if failures:
        raise RuntimeError(f"Inference workers failed: exit codes={failures}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
