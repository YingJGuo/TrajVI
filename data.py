from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(name: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(name))
    ]


class ZipReader:
    _handles = {}

    @classmethod
    def names(cls, path: Path):
        path = str(path)
        if path not in cls._handles:
            cls._handles[path] = zipfile.ZipFile(path, "r")
        return sorted(cls._handles[path].namelist(), key=natural_key)

    @classmethod
    def image(cls, path: Path, name: str):
        path = str(path)
        if path not in cls._handles:
            cls._handles[path] = zipfile.ZipFile(path, "r")
        return Image.open(io.BytesIO(cls._handles[path].read(name)))


def _folder_names(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    names = [
        item.name for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    names.sort(key=natural_key)
    if not names:
        raise FileNotFoundError(f"No images found in {folder}")
    return names


def _resize_frames(frames, size):
    process_size = (int(size[0]) - int(size[0]) % 8,
                    int(size[1]) - int(size[1]) % 8)
    if process_size[0] <= 0 or process_size[1] <= 0:
        raise ValueError(f"Invalid processing size: {process_size}")
    return [frame.resize(process_size) for frame in frames], process_size


def load_frames(data_root: Path, dataset_name: str, video_name: str,
                frame_dir: str, storage_format: str, max_frames: int | None):
    root = data_root / dataset_name / frame_dir
    if storage_format == "zip":
        path = root / f"{video_name}.zip"
        names = ZipReader.names(path)
        if max_frames is not None:
            names = names[:int(max_frames)]
        frames = [ZipReader.image(path, name).convert("RGB") for name in names]
    elif storage_format == "folder":
        folder = root / video_name
        names = _folder_names(folder)
        if max_frames is not None:
            names = names[:int(max_frames)]
        frames = [Image.open(folder / name).convert("RGB") for name in names]
    else:
        raise ValueError("storage_format must be 'zip' or 'folder'")
    if not frames:
        raise ValueError(f"No frames found for {video_name}")
    return frames


def load_masks(data_root: Path, dataset_name: str, video_name: str,
               mask_dir: str, storage_format: str, num_frames: int,
               size, dilation: int):
    root = data_root / dataset_name / mask_dir
    if storage_format == "zip":
        path = root / f"{video_name}.zip"
        names = ZipReader.names(path)[:int(num_frames)]
        if len(names) < int(num_frames):
            raise ValueError(
                f"{path} has {len(names)} masks, expected {num_frames}")
        source = [ZipReader.image(path, name).convert("L") for name in names]
    else:
        folder = root / video_name
        names = _folder_names(folder)[:int(num_frames)]
        if len(names) < int(num_frames):
            raise ValueError(
                f"{folder} has {len(names)} masks, expected {num_frames}")
        source = [Image.open(folder / name).convert("L") for name in names]

    kernel = None
    if int(dilation) > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(dilation), int(dilation)))

    masks = []
    for image in source:
        image = image.resize(size, Image.Resampling.NEAREST)
        mask = (np.asarray(image) > 199).astype(np.uint8)
        if kernel is not None:
            mask = cv2.dilate(mask, kernel, iterations=1)
        masks.append(Image.fromarray(mask * 255))
    return masks


def image_list_to_tensor(images, device):
    array = np.stack([np.asarray(image.convert("RGB")) for image in images])
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor.float().div(255.0).unsqueeze(0).to(device)


def mask_list_to_tensor(images, device):
    array = np.stack([np.asarray(image.convert("L")) for image in images])
    tensor = torch.from_numpy(array).float().div(255.0)
    return tensor[:, None].unsqueeze(0).to(device)


def load_manifest(data_root: Path, dataset_name: str, split: str):
    path = data_root / dataset_name / f"{split}.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if isinstance(manifest, list):
        manifest = {str(name): None for name in manifest}
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a dict/list manifest at {path}")
    return manifest


def get_ref_ids(center: int, local_ids, length: int, stride: int,
                ref_num: int = -1):
    local_ids = set(int(value) for value in local_ids)
    refs = []
    if ref_num == -1:
        candidates = range(0, int(length), int(stride))
    else:
        start = max(0, int(center) - int(stride) * (int(ref_num) // 2))
        end = min(int(length), int(center) + int(stride) * (int(ref_num) // 2))
        candidates = range(start, end, int(stride))
    for frame_id in candidates:
        if frame_id not in local_ids:
            refs.append(int(frame_id))
            if ref_num != -1 and len(refs) >= int(ref_num):
                break
    return refs


def local_ids(center: int, length: int, window: int):
    half = int(window) // 2
    if center < half:
        return list(range(0, min(int(window), int(length))))
    if center + half >= int(length):
        return list(range(max(0, int(length) - int(window)), int(length)))
    return list(range(int(center) - half, int(center) + half))
