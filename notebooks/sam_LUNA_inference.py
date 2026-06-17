#!/usr/bin/env python3
"""
Unified original-SAM inference/evaluation for LUNA-style CT volumes.

This script is for facebookresearch/segment-anything, not SAM2/MedSAM2.
SAM is image-only, so all modes are slice-wise 2D modes:

Modes
-----
1) image-single : Prompted SAM predictor with one mask per prompt.
2) image-multi  : Prompted SAM predictor with multimask output; saves/evaluates ch0/ch1/ch2.
3) auto         : SAM automatic mask generator, merged into one binary slice prediction.

Outputs
-------
- config.yaml with runtime/settings
- predictions.csv with per-volume Dice and status rows
- optional predicted NIfTI volumes under predicted_volumes/<output_key>/

Cluster features
----------------
- deterministic prompt perturbations with --prompt-perturb-rmax
- SLURM-style sharding with --shard-index/--shard-count
- CPU/SimpleITK prefetching with --prefetch-cases
- incremental CSV writing and optional --skip-existing
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import tqdm
from scipy import ndimage

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


PROMPT_MODES = ("point", "box", "point+box")
IMAGE_OUTPUTS = ("point", "box", "point+box", "auto")


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def slugify(value: str, max_len: int = 180) -> str:
    value = str(value).replace("+", "plus")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_.")
    return value[:max_len] if len(value) > max_len else value


def format_float_for_name(value: Optional[float]) -> str:
    if value is None:
        return "all"
    return f"{value:g}".replace(".", "p")


def build_experiment_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == "image-multi":
        output_tag = "out-" + "-".join(m.replace("+", "plus") for m in args.image_outputs if m in PROMPT_MODES)
    elif args.mode == "auto":
        output_tag = "out-auto"
    else:
        output_tag = "out-" + "-".join(m.replace("+", "plus") for m in args.image_outputs)

    parts = [
        timestamp,
        args.run_name,
        "sam",
        args.model_type,
        args.mode,
        output_tag,
        args.amp_dtype,
        "triplet" if args.use_triplet_channels else "singlech",
        "window" if args.use_ct_window else "normslice",
        f"frac-{format_float_for_name(args.dataset_fraction)}",
    ]
    if args.max_cases is not None:
        parts.append(f"max-{args.max_cases}")
    if args.prompt_perturb_rmax > 0:
        parts.append(f"perturb-r{args.prompt_perturb_rmax}")
    if args.mode == "auto" or "auto" in args.image_outputs:
        parts.append(
            f"autoA{args.auto_min_area}-{args.auto_max_area}"
            f"_circ{format_float_for_name(args.auto_min_circularity)}"
            f"_sol{format_float_for_name(args.auto_min_solidity)}"
        )
    if args.shard_count > 1:
        parts.append(f"shard-{args.shard_index}of{args.shard_count}")
    return slugify("_".join(parts))


def namespace_to_yaml_dict(args: argparse.Namespace) -> Dict:
    cfg = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            cfg[key] = str(value)
        elif isinstance(value, (list, tuple)):
            cfg[key] = list(value)
        else:
            cfg[key] = value
    return cfg


def write_yaml_config(config_path: Path, config: Dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        if yaml is not None:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)


def setup_experiment_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.create_experiment_dir:
        exp_name = args.experiment_name or build_experiment_name(args)
        exp_dir = output_root / exp_name
    else:
        exp_dir = output_root
    exp_dir.mkdir(parents=True, exist_ok=args.overwrite_experiment)
    if args.save_volumes:
        (exp_dir / "predicted_volumes").mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(args: argparse.Namespace, exp_dir: Path, device: torch.device, index: Optional["DatasetIndex"] = None) -> None:
    config = {
        "experiment": {
            "experiment_dir": str(exp_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "command": " ".join(os.sys.argv),
        },
        "runtime": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "arguments": namespace_to_yaml_dict(args),
    }
    if index is not None:
        config["dataset_index"] = {
            "dataset_dir": str(index.dataset_dir),
            "volumes_dir": str(index.volumes_dir),
            "masks_dir": str(index.masks_dir),
            "annotations_csv": str(index.annotations_csv),
            "links_csv": str(index.links_csv),
            "n_selected_volumes": len(index.volume_ids),
            "selected_volume_ids": index.volume_ids,
        }
    write_yaml_config(exp_dir / "config.yaml", config)


def dice_score(mask1: np.ndarray, mask2: np.ndarray, smooth: float = 1e-6) -> float:
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum(dtype=np.float64)
    total = mask1.sum(dtype=np.float64) + mask2.sum(dtype=np.float64)
    return float((2.0 * intersection + smooth) / (total + smooth))


def normalize_to_uint8(img_2d: np.ndarray) -> np.ndarray:
    img_2d = img_2d.astype(np.float32)
    img_2d -= float(img_2d.min())
    max_val = float(img_2d.max())
    if max_val > 0:
        img_2d /= max_val
    return (img_2d * 255).astype(np.uint8)


def preprocess_ct_window(image_data: np.ndarray, window_level: float = -750, window_width: float = 1500) -> np.ndarray:
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    out = np.clip(image_data.astype(np.float32), lower, upper)
    denom = float(out.max() - out.min())
    if denom > 0:
        out = (out - float(out.min())) / denom * 255.0
    else:
        out = np.zeros_like(out)
    return out.astype(np.uint8)


def build_sam_input_slice(image_array: np.ndarray, z: int, use_triplet_channels: bool = False) -> np.ndarray:
    """Return an HWC RGB uint8 image for original SAM."""
    if use_triplet_channels:
        z_prev = max(z - 1, 0)
        z_next = min(z + 1, image_array.shape[0] - 1)
        return np.stack(
            [
                normalize_to_uint8(image_array[z_prev]),
                normalize_to_uint8(image_array[z]),
                normalize_to_uint8(image_array[z_next]),
            ],
            axis=-1,
        )
    img = normalize_to_uint8(image_array[z])
    return np.stack([img, img, img], axis=-1)


def safe_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]
    return torch.autocast("cuda", dtype=dtype)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def configure_torch(device: torch.device, allow_tf32: bool) -> None:
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
            torch.set_float32_matmul_precision("high")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"WARNING: ignoring non-integer ${name}={value!r}; using {default}.")
        return default


def stable_uint32_seed(*items) -> int:
    payload = "::".join(str(item) for item in items).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def sample_prompt_shift_xy(base_seed: int, series_id: str, prompt_scope: str, z: int, prompt_index: int, rmax: int) -> Tuple[int, int]:
    rmax = int(rmax)
    if rmax <= 0:
        return 0, 0
    seed = stable_uint32_seed(base_seed, series_id, prompt_scope, z, prompt_index, rmax)
    rng = np.random.default_rng(seed)
    dx, dy = rng.integers(-rmax, rmax + 1, size=2)
    return int(dx), int(dy)


def clip_point_xy(point_xy: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    H, W = image_shape_hw
    point = np.asarray(point_xy, dtype=np.float32).copy()
    point[..., 0] = np.clip(point[..., 0], 0, W - 1)
    point[..., 1] = np.clip(point[..., 1], 0, H - 1)
    return point


def clip_box_xyxy(box_xyxy: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    H, W = image_shape_hw
    box = np.asarray(box_xyxy, dtype=np.float32).copy()
    box[..., 0] = np.clip(box[..., 0], 0, W - 1)
    box[..., 2] = np.clip(box[..., 2], 0, W - 1)
    box[..., 1] = np.clip(box[..., 1], 0, H - 1)
    box[..., 3] = np.clip(box[..., 3], 0, H - 1)
    return box


def pad_box_xyxy(box_xyxy: Sequence[float], image_shape_hw: Tuple[int, int], pad: int) -> np.ndarray:
    H, W = image_shape_hw
    x0, y0, x1, y1 = np.asarray(box_xyxy, dtype=np.float32).reshape(4).tolist()
    return clip_box_xyxy(np.array([x0 - pad, y0 - pad, x1 + pad, y1 + pad], dtype=np.float32), image_shape_hw)


def recenter_box_xyxy_at_point(box_xyxy: np.ndarray, center_xy: Sequence[float], image_shape_hw: Tuple[int, int]) -> np.ndarray:
    """Preserve box size and move center to center_xy as much as image bounds allow."""
    H, W = image_shape_hw
    original = np.asarray(box_xyxy, dtype=np.float32)
    original_shape = original.shape
    x1, y1, x2, y2 = original.reshape(4).tolist()
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    cx, cy = float(center_xy[0]), float(center_xy[1])

    new_x1, new_x2 = cx - box_w / 2.0, cx + box_w / 2.0
    new_y1, new_y2 = cy - box_h / 2.0, cy + box_h / 2.0

    if box_w >= W - 1:
        new_x1, new_x2 = 0.0, float(W - 1)
    else:
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0.0
        if new_x2 > W - 1:
            new_x1 -= new_x2 - (W - 1)
            new_x2 = float(W - 1)
        new_x1 = max(0.0, new_x1)

    if box_h >= H - 1:
        new_y1, new_y2 = 0.0, float(H - 1)
    else:
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0.0
        if new_y2 > H - 1:
            new_y1 -= new_y2 - (H - 1)
            new_y2 = float(H - 1)
        new_y1 = max(0.0, new_y1)

    recentered = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)
    return clip_box_xyxy(recentered, image_shape_hw).reshape(original_shape)


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetIndex:
    dataset_dir: Path
    volumes_dir: Path
    masks_dir: Path
    annotations_csv: Path
    links_csv: Path
    df_annotations: pd.DataFrame
    df_links: pd.DataFrame
    mask_id_to_file: Dict[int, str]
    volume_ids: List[str]


def build_dataset_index(args: argparse.Namespace) -> DatasetIndex:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    volumes_dir = Path(args.volumes_dir).expanduser().resolve() if args.volumes_dir else dataset_dir / "CT_volumes"
    masks_dir = Path(args.masks_dir).expanduser().resolve() if args.masks_dir else dataset_dir / "masks_nodules" / "nifti_data"
    annotations_csv = Path(args.annotations_csv).expanduser().resolve() if args.annotations_csv else dataset_dir / "annotations.csv"
    links_csv = Path(args.links_csv).expanduser().resolve() if args.links_csv else dataset_dir / "LUNA16_metadata_split_offical.csv"

    for p, name in [
        (dataset_dir, "dataset_dir"),
        (volumes_dir, "volumes_dir"),
        (masks_dir, "masks_dir"),
        (annotations_csv, "annotations_csv"),
        (links_csv, "links_csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} does not exist: {p}")

    df_annotations = pd.read_csv(annotations_csv)
    df_links = pd.read_csv(links_csv)
    if "seriesuid" not in df_annotations.columns:
        raise ValueError(f"{annotations_csv} must contain a 'seriesuid' column")
    if not {"SeriesID", "CID"}.issubset(df_links.columns):
        raise ValueError(f"{links_csv} must contain 'SeriesID' and 'CID' columns")

    mask_files = [
        f.name
        for f in masks_dir.iterdir()
        if f.is_file()
        and "mask" in f.name
        and "contour" in f.name
        and "circle" not in f.name
        and "nodule" in f.name
    ]
    mask_id_to_file = {}
    for fname in mask_files:
        try:
            mask_id_to_file[int(fname.split("_")[0])] = fname
        except ValueError:
            continue

    volume_ids = sorted(p.stem for p in volumes_dir.glob("*.mhd"))
    if args.only_annotated:
        annotated = set(df_annotations["seriesuid"].astype(str).tolist())
        volume_ids = [v for v in volume_ids if v in annotated]

    if args.case_list:
        wanted = [line.strip() for line in Path(args.case_list).read_text().splitlines() if line.strip()]
        wanted = [w[:-4] if w.endswith(".mhd") else w for w in wanted]
        wanted_set = set(wanted)
        volume_ids = [v for v in volume_ids if v in wanted_set]

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        volume_ids = list(rng.permutation(volume_ids))

    if args.dataset_fraction is not None:
        if not (0 < args.dataset_fraction <= 1):
            raise ValueError("--dataset-fraction must be in (0, 1]. Use 0.10 for 10%.")
        n = max(1, int(math.ceil(len(volume_ids) * args.dataset_fraction)))
        volume_ids = volume_ids[:n]

    if args.max_cases is not None:
        volume_ids = volume_ids[: max(0, args.max_cases)]

    if args.shard_count > 1:
        if not (0 <= args.shard_index < args.shard_count):
            raise ValueError("--shard-index must be in [0, shard_count).")
        volume_ids = [v for i, v in enumerate(volume_ids) if i % args.shard_count == args.shard_index]

    return DatasetIndex(
        dataset_dir=dataset_dir,
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        annotations_csv=annotations_csv,
        links_csv=links_csv,
        df_annotations=df_annotations,
        df_links=df_links,
        mask_id_to_file=mask_id_to_file,
        volume_ids=volume_ids,
    )


def load_case(index: DatasetIndex, series_id: str, use_ct_window: bool, window_level: float, window_width: float) -> Optional[Tuple[sitk.Image, np.ndarray, np.ndarray, Path]]:
    image_path = index.volumes_dir / f"{series_id}.mhd"
    if not image_path.is_file():
        print(f"Missing image: {image_path}")
        return None

    links = index.df_links[index.df_links["SeriesID"].astype(str) == str(series_id)]
    if len(links) == 0:
        print(f"No mask link found for: {series_id}")
        return None

    mask_id = int(links["CID"].iloc[0])
    mask_fname = index.mask_id_to_file.get(mask_id)
    if mask_fname is None:
        print(f"Mask id not found: CID={mask_id}, series={series_id}")
        return None

    mask_path = index.masks_dir / mask_fname
    if not mask_path.is_file():
        print(f"Missing mask: {mask_path}")
        return None

    image_itk = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image_itk)
    if use_ct_window:
        image_array = preprocess_ct_window(image_array, window_level=window_level, window_width=window_width)

    mask_itk = sitk.ReadImage(str(mask_path))
    gt_volume = (sitk.GetArrayFromImage(mask_itk) >= 0.5).astype(np.uint8)
    if image_array.shape != gt_volume.shape:
        raise ValueError(f"Shape mismatch for {series_id}: image {image_array.shape}, mask {gt_volume.shape}")
    return image_itk, image_array, gt_volume, mask_path


def load_case_safe(index: DatasetIndex, series_id: str, args: argparse.Namespace):
    try:
        return series_id, load_case(index, series_id, args.use_ct_window, args.window_level, args.window_width), None
    except Exception as exc:
        return series_id, None, exc


def iter_loaded_cases(index: DatasetIndex, args: argparse.Namespace):
    ids = list(index.volume_ids)
    prefetch_cases = int(args.prefetch_cases)
    if prefetch_cases <= 0:
        for series_id in ids:
            yield load_case_safe(index, series_id, args)
        return

    max_workers = max(1, prefetch_cases)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = iter(ids)
        futures = {}
        for _ in range(min(max_workers, len(ids))):
            series_id = next(iterator)
            futures[series_id] = executor.submit(load_case_safe, index, series_id, args)
        for series_id in ids:
            fut = futures.pop(series_id)
            try:
                next_series_id = next(iterator)
                futures[next_series_id] = executor.submit(load_case_safe, index, next_series_id, args)
            except StopIteration:
                pass
            yield fut.result()


def write_pred_volume(pred: np.ndarray, reference_image: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    pred_img.CopyInformation(reference_image)
    sitk.WriteImage(pred_img, str(out_path))


# -----------------------------------------------------------------------------
# Prompt extraction and deterministic perturbation
# -----------------------------------------------------------------------------

def get_interior_point(component_mask: np.ndarray) -> List[float]:
    component_mask = (component_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return [float(x), float(y)]


def extract_blobs_from_slice(mask_2d: np.ndarray, pad_box: int = 0) -> List[Dict]:
    mask_bin = (mask_2d > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    H, W = mask_bin.shape
    for contour in contours:
        if contour is None or len(contour) == 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        component_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], contourIdx=-1, color=1, thickness=-1)
        if component_mask.sum() == 0:
            continue
        bbox = [int(x), int(y), int(x + w - 1), int(y + h - 1)]
        if pad_box > 0:
            bbox = pad_box_xyxy(bbox, image_shape_hw=(H, W), pad=pad_box).astype(np.float32).tolist()
        blobs.append(
            {
                "center": get_interior_point(component_mask),
                "bbox": bbox,
                "component_mask": component_mask,
                "contour": contour,
            }
        )
    blobs.sort(key=lambda b: (b["bbox"][1], b["bbox"][0], b["bbox"][3], b["bbox"][2]))
    return blobs


def build_slice_prompt_dict(mask_3d: np.ndarray, pad_box: int = 0) -> List[Dict]:
    mask_3d = (mask_3d > 0).astype(np.uint8)
    slice_dicts = []
    for z in range(mask_3d.shape[0]):
        mask_slice = mask_3d[z]
        if not np.any(mask_slice > 0):
            continue
        blobs = extract_blobs_from_slice(mask_slice, pad_box=pad_box)
        if not blobs:
            continue
        slice_dicts.append(
            {
                "z": int(z),
                "mask_slice": mask_slice,
                "blobs": blobs,
                "point_coords": np.array([b["center"] for b in blobs], dtype=np.float32),
                "point_labels": np.ones(len(blobs), dtype=np.int32),
                "boxes": np.array([b["bbox"] for b in blobs], dtype=np.float32),
            }
        )
    return slice_dicts


def perturb_slice_prompt_dicts(slice_dicts: List[Dict], series_id: str, image_shape_hw: Tuple[int, int], rmax: int, seed: int) -> List[Dict]:
    rmax = int(rmax)
    if rmax <= 0:
        return slice_dicts

    perturbed = []
    for info in slice_dicts:
        new_info = dict(info)
        z = int(info["z"])
        points = np.asarray(info["point_coords"], dtype=np.float32).copy()
        boxes = np.asarray(info["boxes"], dtype=np.float32).copy()
        shifts = []
        for i in range(points.shape[0]):
            dx, dy = sample_prompt_shift_xy(seed, series_id, "sam-image-slice", z, i, rmax)
            shifted = clip_point_xy(points[i] + np.array([dx, dy], dtype=np.float32), image_shape_hw)
            points[i] = shifted
            boxes[i] = recenter_box_xyxy_at_point(boxes[i], shifted, image_shape_hw)
            shifts.append([dx, dy])
        new_info["point_coords"] = points
        new_info["boxes"] = boxes
        new_info["perturb_shifts_xy"] = np.array(shifts, dtype=np.int32)
        perturbed.append(new_info)
    return perturbed


# -----------------------------------------------------------------------------
# Automatic mask filtering
# -----------------------------------------------------------------------------

def compute_mask_shape_features(mask: np.ndarray) -> Optional[Dict]:
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    if area == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else np.inf
    elongation = max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-8))
    circularity = 4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else 0.0
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0
    return {
        "area": area,
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(aspect_ratio),
        "elongation": float(elongation),
    }


def is_nodule_like_mask(mask: np.ndarray, min_area: int, max_area: int, min_circularity: float, min_solidity: float, max_elongation: float) -> Tuple[bool, Optional[Dict]]:
    feats = compute_mask_shape_features(mask)
    if feats is None:
        return False, None
    keep = (
        feats["area"] >= min_area
        and feats["area"] <= max_area
        and feats["circularity"] >= min_circularity
        and feats["solidity"] >= min_solidity
        and feats["elongation"] <= max_elongation
    )
    return keep, feats


def merge_automatic_masks_blob_like(anns: List[Dict], image_shape: Tuple[int, int], args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, float]]:
    H, W = image_shape
    merged = np.zeros((H, W), dtype=bool)
    n_total = 0
    n_after_quality = 0
    n_kept = 0

    for ann in anns:
        n_total += 1
        seg = ann["segmentation"].astype(np.uint8)
        pred_iou = ann.get("predicted_iou")
        stability = ann.get("stability_score")
        if args.auto_filter_pred_iou_thresh is not None and pred_iou is not None and pred_iou < args.auto_filter_pred_iou_thresh:
            continue
        if args.auto_filter_stability_score_thresh is not None and stability is not None and stability < args.auto_filter_stability_score_thresh:
            continue
        n_after_quality += 1
        keep, _ = is_nodule_like_mask(
            seg,
            min_area=args.auto_min_area,
            max_area=args.auto_max_area,
            min_circularity=args.auto_min_circularity,
            min_solidity=args.auto_min_solidity,
            max_elongation=args.auto_max_elongation,
        )
        if keep:
            merged |= seg.astype(bool)
            n_kept += 1

    stats = {
        "auto_n_total_masks": float(n_total),
        "auto_n_after_quality": float(n_after_quality),
        "auto_n_kept_masks": float(n_kept),
    }
    return merged.astype(np.uint8), stats


# -----------------------------------------------------------------------------
# SAM builders and predictors
# -----------------------------------------------------------------------------

def build_sam_model(args: argparse.Namespace, device: torch.device):
    if args.model_type not in sam_model_registry:
        raise ValueError(f"Unknown --model-type {args.model_type!r}. Available: {sorted(sam_model_registry.keys())}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam.eval()
    return sam


def build_sam_predictor(sam_model) -> SamPredictor:
    return SamPredictor(sam_model)


def build_sam_mask_generator(sam_model, args: argparse.Namespace) -> SamAutomaticMaskGenerator:
    return SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=args.auto_points_per_side,
        points_per_batch=args.auto_points_per_batch,
        pred_iou_thresh=args.auto_generator_pred_iou_thresh,
        stability_score_thresh=args.auto_generator_stability_score_thresh,
        stability_score_offset=args.auto_stability_score_offset,
        box_nms_thresh=args.auto_box_nms_thresh,
        crop_n_layers=args.auto_crop_n_layers,
        crop_nms_thresh=args.auto_crop_nms_thresh,
        crop_overlap_ratio=args.auto_crop_overlap_ratio,
        crop_n_points_downscale_factor=args.auto_crop_n_points_downscale_factor,
        min_mask_region_area=args.auto_min_mask_region_area,
        output_mode="binary_mask",
    )


# -----------------------------------------------------------------------------
# Image modes
# -----------------------------------------------------------------------------

def predict_image_single(
    series_id: str,
    image_array: np.ndarray,
    gt_volume: np.ndarray,
    predictor: Optional[SamPredictor],
    mask_generator: Optional[SamAutomaticMaskGenerator],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    prompts = build_slice_prompt_dict(gt_volume, pad_box=args.image_pad_box)
    prompts = perturb_slice_prompt_dicts(
        prompts,
        series_id=series_id,
        image_shape_hw=gt_volume.shape[1:],
        rmax=args.prompt_perturb_rmax,
        seed=args.seed,
    )
    if not prompts:
        raise ValueError("No valid GT-positive prompt slices found")

    requested = set(args.image_outputs)
    need_auto = "auto" in requested
    need_prompted = any(m in requested for m in PROMPT_MODES)
    prompt_by_z = {int(d["z"]): d for d in prompts}

    if args.process_all_slices_auto:
        auto_z_list = list(range(image_array.shape[0]))
    else:
        auto_z_list = sorted(prompt_by_z.keys())

    z_to_process = set()
    if need_auto:
        z_to_process.update(auto_z_list)
    if need_prompted:
        z_to_process.update(prompt_by_z.keys())

    pred = {mode: np.zeros_like(gt_volume, dtype=bool) for mode in requested}
    scores = {mode: [] for mode in PROMPT_MODES}
    auto_stats_accum = {"auto_n_total_masks": 0.0, "auto_n_after_quality": 0.0, "auto_n_kept_masks": 0.0}

    for z in sorted(z_to_process):
        if need_auto and z in auto_z_list:
            if mask_generator is None:
                raise RuntimeError("Automatic output requested but mask_generator is None")
            img_auto = build_sam_input_slice(image_array, z, use_triplet_channels=args.use_triplet_channels)
            with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
                anns = mask_generator.generate(img_auto)
            pred_slice, stats = merge_automatic_masks_blob_like(anns, img_auto.shape[:2], args)
            pred["auto"][z] |= pred_slice.astype(bool)
            for k, v in stats.items():
                auto_stats_accum[k] += float(v)

        if need_prompted and z in prompt_by_z:
            if predictor is None:
                raise RuntimeError("Prompted output requested but predictor is None")
            info = prompt_by_z[z]
            img = build_sam_input_slice(image_array, z, use_triplet_channels=args.use_triplet_channels)
            with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
                predictor.set_image(img, image_format="RGB")
                slice_acc = {m: np.zeros_like(gt_volume[z], dtype=bool) for m in requested if m in PROMPT_MODES}
                for i in range(len(info["blobs"])):
                    point = info["point_coords"][i : i + 1]
                    label = info["point_labels"][i : i + 1]
                    box = info["boxes"][i]
                    if "point" in requested:
                        masks, sc, _ = predictor.predict(point_coords=point, point_labels=label, multimask_output=False)
                        slice_acc["point"] |= masks[0].astype(bool)
                        scores["point"].append(float(sc[0]))
                    if "box" in requested:
                        masks, sc, _ = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=False)
                        slice_acc["box"] |= masks[0].astype(bool)
                        scores["box"].append(float(sc[0]))
                    if "point+box" in requested:
                        masks, sc, _ = predictor.predict(point_coords=point, point_labels=label, box=box, multimask_output=False)
                        slice_acc["point+box"] |= masks[0].astype(bool)
                        scores["point+box"].append(float(sc[0]))
                for mode, sl in slice_acc.items():
                    pred[mode][z] |= sl
                predictor.reset_image()

    pred = {k: v.astype(np.uint8) for k, v in pred.items()}
    row = {"VolumeID": series_id, "n_prompt_slices": len(prompts)}
    for mode in args.image_outputs:
        if mode in pred:
            row[f"DSC ({mode})"] = dice_score(gt_volume, pred[mode])
    for mode in PROMPT_MODES:
        if scores[mode]:
            row[f"score_mean ({mode})"] = float(np.mean(scores[mode]))
            row[f"score_n ({mode})"] = len(scores[mode])
    if need_auto:
        row.update(auto_stats_accum)
    return row, pred


def predict_image_multi(
    series_id: str,
    image_array: np.ndarray,
    gt_volume: np.ndarray,
    predictor: SamPredictor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    prompts = build_slice_prompt_dict(gt_volume, pad_box=args.image_pad_box)
    prompts = perturb_slice_prompt_dicts(
        prompts,
        series_id=series_id,
        image_shape_hw=gt_volume.shape[1:],
        rmax=args.prompt_perturb_rmax,
        seed=args.seed,
    )
    if not prompts:
        raise ValueError("No valid GT-positive prompt slices found")

    requested = [m for m in args.image_outputs if m in PROMPT_MODES]
    if not requested:
        raise ValueError("image-multi requires at least one prompted output: point, box, or point+box")

    pred = {f"{mode}_ch{ch}": np.zeros_like(gt_volume, dtype=bool) for mode in requested for ch in range(3)}
    scores = {f"{mode}_ch{ch}": [] for mode in requested for ch in range(3)}

    for info in prompts:
        z = int(info["z"])
        img = build_sam_input_slice(image_array, z, use_triplet_channels=args.use_triplet_channels)
        with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
            predictor.set_image(img, image_format="RGB")
            slice_acc = {k: np.zeros_like(gt_volume[z], dtype=bool) for k in pred}
            for i in range(len(info["blobs"])):
                point = info["point_coords"][i : i + 1]
                label = info["point_labels"][i : i + 1]
                box = info["boxes"][i]
                outputs = {}
                if "point" in requested:
                    outputs["point"] = predictor.predict(point_coords=point, point_labels=label, multimask_output=True)[:2]
                if "box" in requested:
                    outputs["box"] = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=True)[:2]
                if "point+box" in requested:
                    outputs["point+box"] = predictor.predict(point_coords=point, point_labels=label, box=box, multimask_output=True)[:2]
                for mode, (masks, sc) in outputs.items():
                    n_masks = min(3, masks.shape[0])
                    for ch in range(n_masks):
                        key = f"{mode}_ch{ch}"
                        slice_acc[key] |= masks[ch].astype(bool)
                        scores[key].append(float(sc[ch]))
            for key, sl in slice_acc.items():
                pred[key][z] |= sl
            predictor.reset_image()

    pred = {k: v.astype(np.uint8) for k, v in pred.items()}
    row = {"VolumeID": series_id, "n_prompt_slices": len(prompts)}
    for key, vol in pred.items():
        row[f"DSC ({key})"] = dice_score(gt_volume, vol)
        if scores[key]:
            row[f"score_mean ({key})"] = float(np.mean(scores[key]))
            row[f"score_n ({key})"] = len(scores[key])
    return row, pred


def predict_auto(
    series_id: str,
    image_array: np.ndarray,
    gt_volume: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    if args.process_all_slices_auto:
        z_list = list(range(image_array.shape[0]))
        n_prompt_slices = int(np.any(gt_volume > 0, axis=(1, 2)).sum())
    else:
        prompts = build_slice_prompt_dict(gt_volume, pad_box=args.image_pad_box)
        if not prompts:
            raise ValueError("No valid GT-positive prompt slices found")
        z_list = sorted(int(d["z"]) for d in prompts)
        n_prompt_slices = len(z_list)

    pred_auto = np.zeros_like(gt_volume, dtype=np.uint8)
    auto_stats_accum = {"auto_n_total_masks": 0.0, "auto_n_after_quality": 0.0, "auto_n_kept_masks": 0.0}

    for z in z_list:
        img = build_sam_input_slice(image_array, z, use_triplet_channels=args.use_triplet_channels)
        with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
            anns = mask_generator.generate(img)
        pred_slice, stats = merge_automatic_masks_blob_like(anns, img.shape[:2], args)
        pred_auto[z] = pred_slice.astype(np.uint8)
        for k, v in stats.items():
            auto_stats_accum[k] += float(v)

    row = {"VolumeID": series_id, "n_auto_slices": len(z_list), "n_prompt_slices": n_prompt_slices}
    row["DSC (auto)"] = dice_score(gt_volume, pred_auto)
    row.update(auto_stats_accum)
    return row, {"auto": pred_auto.astype(np.uint8)}


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original SAM slice-wise LUNA evaluation with deterministic prompt perturbations.")

    # Dataset paths
    parser.add_argument("--dataset-dir", required=True, help="Dataset root. Defaults inside assume LUNA-style folders.")
    parser.add_argument("--volumes-dir", default=None, help="Override CT volumes directory. Default: dataset_dir/CT_volumes")
    parser.add_argument("--masks-dir", default=None, help="Override masks directory. Default: dataset_dir/masks_nodules/nifti_data")
    parser.add_argument("--annotations-csv", default=None, help="Default: dataset_dir/annotations.csv")
    parser.add_argument("--links-csv", default=None, help="Default: dataset_dir/LUNA16_metadata_split_offical.csv")

    # Output paths
    parser.add_argument("--output-dir", default="results_sam", help="Output root.")
    parser.add_argument("--run-name", default="luna_sam", help="Short name embedded in the experiment folder.")

    # Model
    parser.add_argument("--model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"], help="SAM model type for sam_model_registry.")
    parser.add_argument("--checkpoint", required=True, help="SAM checkpoint path, e.g. sam_vit_h_4b8939.pth")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16", help="CUDA autocast dtype. Use bf16 on H100/H200; use none if you see numerical issues.")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 for CUDA matmul/cuDNN.")

    # Mode selection
    parser.add_argument("--mode", choices=["image-single", "image-multi", "auto"], required=True)
    parser.add_argument(
        "--image-outputs",
        nargs="+",
        choices=IMAGE_OUTPUTS,
        default=["point", "box", "point+box"],
        help="Outputs for image-single. For image-multi only point/box/point+box are used. For auto mode this is ignored.",
    )
    parser.add_argument("--use-triplet-channels", action="store_true", help="Use z-1/z/z+1 as RGB channels instead of duplicated slice.")
    parser.add_argument("--use-ct-window", action="store_true", help="Window the whole CT volume before slice normalization.")
    parser.add_argument("--window-level", type=float, default=-750)
    parser.add_argument("--window-width", type=float, default=1500)
    parser.add_argument(
        "--prompt-perturb-rmax",
        type=int,
        default=0,
        help=(
            "Maximum absolute deterministic random perturbation in pixels for prompted point/box inputs. "
            "For each prompt, dx and dy are sampled from [-rmax, +rmax]. The shifted point and "
            "recentered box are shared by point, box, and point+box modes. Use 0 to disable."
        ),
    )
    parser.add_argument("--image-pad-box", type=int, default=0, help="Padding in pixels around slice-wise prompt boxes.")

    # Automatic mask generator options
    parser.add_argument("--process-all-slices-auto", action="store_true", help="Run automatic mask generation on all CT slices. Default is only GT-positive slices to save time.")
    parser.add_argument("--auto-points-per-side", type=int, default=32)
    parser.add_argument("--auto-points-per-batch", type=int, default=64)
    parser.add_argument("--auto-generator-pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--auto-generator-stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--auto-stability-score-offset", type=float, default=1.0)
    parser.add_argument("--auto-box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--auto-crop-n-layers", type=int, default=0)
    parser.add_argument("--auto-crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--auto-crop-overlap-ratio", type=float, default=512 / 1500)
    parser.add_argument("--auto-crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--auto-min-mask-region-area", type=int, default=0, help="SAM postprocessing min area. Requires OpenCV.")

    # Extra nodule-like filtering after SAM AMG
    parser.add_argument("--auto-min-area", type=int, default=3)
    parser.add_argument("--auto-max-area", type=int, default=5000)
    parser.add_argument("--auto-min-circularity", type=float, default=0.0)
    parser.add_argument("--auto-min-solidity", type=float, default=0.0)
    parser.add_argument("--auto-max-elongation", type=float, default=100.0)
    parser.add_argument("--auto-filter-pred-iou-thresh", type=float, default=None, help="Additional post-generate predicted_iou filter; None disables.")
    parser.add_argument("--auto-filter-stability-score-thresh", type=float, default=None, help="Additional post-generate stability filter; None disables.")

    # Dataset selection / cluster efficiency
    parser.add_argument("--only-annotated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-fraction", type=float, default=None, help="Analyze a fraction of selected dataset, e.g. 0.10 for 10%.")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap number of volumes after filtering/subsampling.")
    parser.add_argument("--case-list", default=None, help="Text file with one SeriesID per line.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before fraction/max/sharding selection.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--shard-index",
        nargs="?",
        type=int,
        const=env_int("SLURM_ARRAY_TASK_ID", 0),
        default=env_int("SLURM_ARRAY_TASK_ID", 0),
        help="Shard index. If provided without a value, uses SLURM_ARRAY_TASK_ID or 0.",
    )
    parser.add_argument(
        "--shard-count",
        nargs="?",
        type=int,
        const=env_int("SLURM_ARRAY_TASK_COUNT", 1),
        default=env_int("SLURM_ARRAY_TASK_COUNT", 1),
        help="Number of shards. If provided without a value, uses SLURM_ARRAY_TASK_COUNT or 1.",
    )
    parser.add_argument("--prefetch-cases", type=int, default=0, help="CPU/SimpleITK case prefetch workers. Try 1-2 on a cluster.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases whose requested prediction volumes already exist.")
    parser.add_argument("--empty-cache-every", type=int, default=10, help="Call torch.cuda.empty_cache every N cases; 0 disables.")

    # Saving
    parser.add_argument("--save-volumes", action="store_true", help="Also save predicted NIfTI volumes. If false, only CSV is saved.")
    parser.add_argument("--save-gt-volume", action="store_true", help="Save GT mask next to predictions for inspection.")
    parser.add_argument("--csv-filename", default=None, help="Override CSV file name.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise errors immediately instead of recording them in CSV.")
    parser.add_argument(
        "--create-experiment-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a timestamped experiment folder inside --output-dir.",
    )
    parser.add_argument("--experiment-name", default=None, help="Optional custom experiment folder name.")
    parser.add_argument("--overwrite-experiment", action="store_true", help="Allow writing into an existing experiment folder.")

    args = parser.parse_args()
    if args.prompt_perturb_rmax < 0:
        raise ValueError("--prompt-perturb-rmax must be >= 0")
    if args.image_pad_box < 0:
        raise ValueError("--image-pad-box must be >= 0")
    if args.mode == "image-multi":
        args.image_outputs = [m for m in args.image_outputs if m in PROMPT_MODES]
        if not args.image_outputs:
            args.image_outputs = list(PROMPT_MODES)
    if args.mode == "auto":
        args.image_outputs = ["auto"]
    return args


def expected_volume_keys(args: argparse.Namespace) -> List[str]:
    if args.mode == "image-single":
        return [m.replace("+", "_") for m in args.image_outputs]
    if args.mode == "image-multi":
        return [f"{m.replace('+', '_')}_ch{ch}" for m in args.image_outputs for ch in range(3)]
    if args.mode == "auto":
        return ["auto"]
    raise ValueError(args.mode)


def expected_volume_paths(out_dir: Path, series_id: str, keys: Sequence[str]) -> List[Path]:
    return [out_dir / "predicted_volumes" / key / f"{series_id}_{key}.nii.gz" for key in keys]


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    configure_torch(device=device, allow_tf32=args.allow_tf32)

    out_dir = setup_experiment_dir(args)
    print(f"Using device: {device}; mode={args.mode}; model_type={args.model_type}; amp={args.amp_dtype}; tf32={args.allow_tf32}")
    print(f"Experiment directory: {out_dir}")

    index = build_dataset_index(args)
    save_experiment_config(args=args, exp_dir=out_dir, device=device, index=index)
    print(f"Saved config: {out_dir / 'config.yaml'}")
    print(f"Selected {len(index.volume_ids)} volumes for this run/shard.")
    print(f"Shard index/count: {args.shard_index}/{args.shard_count}; prefetch_cases={args.prefetch_cases}")

    sam_model = build_sam_model(args, device)
    predictor = None
    mask_generator = None
    if args.mode in {"image-single", "image-multi"} and any(m in args.image_outputs for m in PROMPT_MODES):
        predictor = build_sam_predictor(sam_model)
    if args.mode == "auto" or (args.mode == "image-single" and "auto" in args.image_outputs):
        mask_generator = build_sam_mask_generator(sam_model, args)

    rows: List[Dict] = []
    csv_name = args.csv_filename or "predictions.csv"
    csv_path = out_dir / csv_name
    volume_keys = expected_volume_keys(args)

    case_iter = iter_loaded_cases(index, args)
    for case_idx, (series_id, loaded, load_error) in enumerate(
        tqdm.tqdm(case_iter, total=len(index.volume_ids), desc="Volumes"),
        start=1,
    ):
        try:
            if load_error is not None:
                raise load_error
            if loaded is None:
                rows.append({"VolumeID": series_id, "status": "missing_input"})
                continue
            image_itk, image_array, gt_volume, mask_path = loaded

            if args.skip_existing and args.save_volumes:
                paths = expected_volume_paths(out_dir, series_id, volume_keys)
                if all(p.exists() for p in paths):
                    rows.append({"VolumeID": series_id, "status": "skipped_existing"})
                    continue

            if args.mode == "image-single":
                row, pred_volumes = predict_image_single(series_id, image_array, gt_volume, predictor, mask_generator, args, device)
            elif args.mode == "image-multi":
                row, pred_volumes = predict_image_multi(series_id, image_array, gt_volume, predictor, args, device)
            elif args.mode == "auto":
                row, pred_volumes = predict_auto(series_id, image_array, gt_volume, mask_generator, args, device)
            else:
                raise ValueError(args.mode)

            row.update(
                {
                    "status": "ok",
                    "mask_file": Path(mask_path).name,
                    "n_slices": int(gt_volume.shape[0]),
                    "height": int(gt_volume.shape[1]),
                    "width": int(gt_volume.shape[2]),
                    "prompt_perturb_rmax": int(args.prompt_perturb_rmax),
                    "prompt_perturb_seed": int(args.seed),
                    "model_type": str(args.model_type),
                }
            )
            rows.append(row)

            if args.save_volumes:
                pred_root = out_dir / "predicted_volumes"
                for key, vol in pred_volumes.items():
                    safe_key = key.replace("+", "_")
                    write_pred_volume(vol, image_itk, pred_root / safe_key / f"{series_id}_{safe_key}.nii.gz")
                if args.save_gt_volume:
                    write_pred_volume(gt_volume, image_itk, pred_root / "gt" / f"{series_id}_gt.nii.gz")

        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append({"VolumeID": series_id, "status": "error", "error": repr(exc)})
            print(f"ERROR in {series_id}: {repr(exc)}")

        if args.empty_cache_every and case_idx % args.empty_cache_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    dice_cols = [c for c in df.columns if c.startswith("DSC")]
    for c in dice_cols:
        print(f"Mean {c}: {pd.to_numeric(df[c], errors='coerce').mean():.6f}")


if __name__ == "__main__":
    main()