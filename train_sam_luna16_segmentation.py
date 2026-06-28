#!/usr/bin/env python3
"""
Fully supervised positive-slice nodule segmentation with original SAM encoder.

This script is for facebookresearch/segment-anything, not SAM2/MedSAM2.
It trains a promptless binary segmentation model on LUNA-style nodule-positive
2D/2.5D slices:

    CT slice/triplet -> original SAM image encoder -> selectable CNN decoder/head -> mask logits

The SAM prompt encoder and SAM mask decoder are not used. This is intentional:
it gives a simple fully-supervised baseline while still using SAM image features.

Dataset assumptions match the accompanying LUNA SAM inference script:
    DATASET_DIR/CT_volumes/*.mhd
    DATASET_DIR/masks_nodules/nifti_data/*mask*contour*nodule*.nii.gz
    DATASET_DIR/annotations.csv
    DATASET_DIR/LUNA16_metadata_split_offical.csv with columns SeriesID,CID

Outputs
-------
- timestamped experiment directory by default
- config.yaml
- splits/train.txt, val.txt, test.txt, all.txt
- metrics.csv / metrics.json
- best_model.pt / last_model.pt
- test_predictions/{best_model,last_model}/
    - slice_metrics.csv
    - patient_slice_summary.csv
    - patient_volume_metrics.csv
    - nodule_metrics.csv
    - patient_nodule_summary.csv
    - summary.csv / summary.json
    - predicted_volumes/*.nii.gz

Example
-------
python train_sam_luna_positive_slice_segmentation.py \
  --dataset-dir /path/to/LUNA16 \
  --model-type vit_h \
  --checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --output-dir results/sam_positive_slice_seg \
  --epochs 30 --batch-size 4 --eval-batch-size 8 \
  --use-triplet-channels --amp-dtype bf16 --save-volumes
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import re
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from segment_anything import sam_model_registry


# =============================================================================
# General utilities
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(value: str, max_len: int = 180) -> str:
    value = str(value).replace("+", "plus")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_.")
    return value[:max_len]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def configure_torch(device: torch.device, allow_tf32: bool = True) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
            torch.set_float32_matmul_precision("high")


def safe_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]
    return torch.autocast("cuda", dtype=dtype)


def to_plain_python(obj):
    """Recursively convert NumPy/Pandas/Path objects into YAML/JSON-safe Python types."""

    # Handle NumPy scalars before Python primitive checks.
    # In particular, np.str_ can behave like str for isinstance(), but PyYAML
    # still cannot serialize it unless it is converted to a native Python str.
    if isinstance(obj, np.generic):
        return to_plain_python(obj.item())

    if isinstance(obj, np.ndarray):
        return [to_plain_python(v) for v in obj.tolist()]

    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, argparse.Namespace):
        return {str(k): to_plain_python(v) for k, v in vars(obj).items()}

    if isinstance(obj, dict):
        return {str(to_plain_python(k)): to_plain_python(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_plain_python(v) for v in obj]

    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else float(obj)

    if isinstance(obj, (str, int, bool)):
        return obj

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)


def namespace_to_plain_dict(args: argparse.Namespace) -> Dict:
    return to_plain_python(args)


def write_config(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_plain_python(payload)
    with open(path, "w") as f:
        if yaml is not None:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        else:
            json.dump(payload, f, indent=2, allow_nan=False)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_plain_python(payload), f, indent=2, allow_nan=False)


def write_pred_volume(pred: np.ndarray, reference_image: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(pred.astype(np.uint8))
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, str(out_path))


def get_spacing_zyx(image_itk: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = image_itk.GetSpacing()
    return float(sz), float(sy), float(sx)


# =============================================================================
# Metrics
# =============================================================================


def binary_confusion_counts(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = int(np.logical_and(gt_b, pred_b).sum())
    fp = int(np.logical_and(~gt_b, pred_b).sum())
    fn = int(np.logical_and(gt_b, ~pred_b).sum())
    tn = int(np.logical_and(~gt_b, ~pred_b).sum())
    return tp, fp, fn, tn


def dice_score(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    inter = np.logical_and(gt_b, pred_b).sum(dtype=np.float64)
    total = gt_b.sum(dtype=np.float64) + pred_b.sum(dtype=np.float64)
    return float((2.0 * inter + smooth) / (total + smooth))


def iou_score(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    inter = np.logical_and(gt_b, pred_b).sum(dtype=np.float64)
    union = np.logical_or(gt_b, pred_b).sum(dtype=np.float64)
    return float((inter + smooth) / (union + smooth))


def precision_recall_f1(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    tp, fp, fn, tn = binary_confusion_counts(gt, pred)
    precision = float((tp + smooth) / (tp + fp + smooth))
    recall = float((tp + smooth) / (tp + fn + smooth))
    f1 = float((2.0 * precision * recall + smooth) / (precision + recall + smooth))
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)


def surface_distances_mm(mask_a: np.ndarray, mask_b: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() or not b.any():
        return np.array([], dtype=np.float32)
    surf_a = _surface(a)
    surf_b = _surface(b)
    dt_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing_zyx)
    dt_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing_zyx)
    return np.concatenate([dt_b[surf_a], dt_a[surf_b]]).astype(np.float32)


def hd95_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances_mm(gt, pred, spacing_zyx)
    return float(np.percentile(d, 95)) if d.size else float("nan")


def assd_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances_mm(gt, pred, spacing_zyx)
    return float(np.mean(d)) if d.size else float("nan")


def volume_similarity(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    g = float(gt.astype(bool).sum())
    p = float(pred.astype(bool).sum())
    return float(1.0 - abs(p - g) / (p + g + smooth))


def segmentation_metrics(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Optional[Tuple[float, float, float]] = None) -> Dict[str, float]:
    pr = precision_recall_f1(gt, pred)
    out = {
        "Dice": dice_score(gt, pred),
        "IoU": iou_score(gt, pred),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "f1": pr["f1"],
        "tp": pr["tp"],
        "fp": pr["fp"],
        "fn": pr["fn"],
        "tn": pr["tn"],
        "pred_voxels": int(pred.astype(bool).sum()),
        "gt_voxels": int(gt.astype(bool).sum()),
        "volume_similarity": volume_similarity(gt, pred),
    }
    if spacing_zyx is not None:
        out["HD95_mm"] = hd95_mm(gt, pred, spacing_zyx)
        out["ASSD_mm"] = assd_mm(gt, pred, spacing_zyx)
    return out


# =============================================================================
# CT preprocessing
# =============================================================================


def normalize_to_uint8(img_2d: np.ndarray) -> np.ndarray:
    img = img_2d.astype(np.float32)
    img -= float(img.min())
    max_val = float(img.max())
    if max_val > 0:
        img /= max_val
    return np.round(img * 255.0).astype(np.uint8)


def normalize_ct_to_uint8(img_2d: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    img = img_2d.astype(np.float32)
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / max(hu_max - hu_min, 1e-6)
    return np.round(img * 255.0).astype(np.uint8)


def build_sam_input_slice(
    image_array: np.ndarray,
    z: int,
    use_triplet_channels: bool,
    use_ct_window: bool,
    hu_min: float,
    hu_max: float,
) -> np.ndarray:
    """Return H,W,3 uint8 input for original SAM."""
    def norm(k: int) -> np.ndarray:
        if use_ct_window:
            return normalize_ct_to_uint8(image_array[k], hu_min=hu_min, hu_max=hu_max)
        return normalize_to_uint8(image_array[k])

    if use_triplet_channels:
        z0 = max(z - 1, 0)
        z1 = z
        z2 = min(z + 1, image_array.shape[0] - 1)
        return np.stack([norm(z0), norm(z1), norm(z2)], axis=-1)
    ch = norm(z)
    return np.stack([ch, ch, ch], axis=-1)


def resize_rgb_and_mask(rgb: np.ndarray, mask: np.ndarray, image_size: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Optionally resize a H,W,3 uint8 image and H,W binary mask to a square size."""
    if image_size is None:
        return rgb, mask
    size = (int(image_size), int(image_size))
    rgb_r = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return rgb_r, mask_r


def resize_rgb_only(rgb: np.ndarray, image_size: Optional[int]) -> np.ndarray:
    """Optionally resize a H,W,3 uint8 image to a square size."""
    if image_size is None:
        return rgb
    size = (int(image_size), int(image_size))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)


def build_augment_params(args: argparse.Namespace) -> Dict:
    return {
        "enabled": bool(args.augment),
        "hflip_p": float(args.aug_hflip_p),
        "vflip_p": float(args.aug_vflip_p),
        "rotation_deg": float(args.aug_rotation_deg),
        "shift_px": float(args.aug_shift_px),
        "scale_min": float(args.aug_scale_min),
        "scale_max": float(args.aug_scale_max),
        "intensity_p": float(args.aug_intensity_p),
        "brightness": float(args.aug_brightness),
        "contrast": float(args.aug_contrast),
        "noise_std": float(args.aug_noise_std),
        "blur_p": float(args.aug_blur_p),
    }


def apply_train_augmentations(rgb: np.ndarray, mask: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Apply conservative 2D augmentations to a single training slice/mask pair.

    Geometric transforms are shared by image and mask. Intensity transforms are
    applied only to the image. The image remains uint8 H,W,3 and the mask remains
    uint8 H,W. Augmentation is intentionally mild because LUNA nodules are small.
    """
    if not params or not params.get("enabled", False):
        return rgb, mask

    rgb = np.ascontiguousarray(rgb)
    mask = np.ascontiguousarray(mask.astype(np.uint8))

    if random.random() < params.get("hflip_p", 0.0):
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < params.get("vflip_p", 0.0):
        rgb = np.ascontiguousarray(rgb[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])

    H, W = mask.shape[:2]
    rot = float(params.get("rotation_deg", 0.0))
    shift = float(params.get("shift_px", 0.0))
    scale_min = float(params.get("scale_min", 1.0))
    scale_max = float(params.get("scale_max", 1.0))
    if rot > 0 or shift > 0 or abs(scale_min - 1.0) > 1e-6 or abs(scale_max - 1.0) > 1e-6:
        angle = random.uniform(-rot, rot) if rot > 0 else 0.0
        scale = random.uniform(scale_min, scale_max) if scale_max > 0 else 1.0
        tx = random.uniform(-shift, shift) if shift > 0 else 0.0
        ty = random.uniform(-shift, shift) if shift > 0 else 0.0
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        rgb = cv2.warpAffine(
            rgb, M, (W, H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask, M, (W, H), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    if random.random() < params.get("intensity_p", 0.0):
        x = rgb.astype(np.float32)
        contrast = float(params.get("contrast", 0.0))
        brightness = float(params.get("brightness", 0.0))
        if contrast > 0:
            x *= random.uniform(1.0 - contrast, 1.0 + contrast)
        if brightness > 0:
            x += random.uniform(-brightness, brightness) * 255.0
        noise_std = float(params.get("noise_std", 0.0))
        if noise_std > 0:
            x += np.random.normal(0.0, noise_std * 255.0, size=x.shape).astype(np.float32)
        rgb = np.clip(x, 0.0, 255.0).astype(np.uint8)

    if random.random() < params.get("blur_p", 0.0):
        rgb = cv2.GaussianBlur(rgb, ksize=(3, 3), sigmaX=0.0)

    return rgb.astype(np.uint8), (mask > 0).astype(np.uint8)


def numpy_rgb_to_sam_tensor(rgb: np.ndarray) -> torch.Tensor:
    """H,W,3 uint8 -> 3,H,W float in [0,255]."""
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


# =============================================================================
# Dataset indexing/loading
# =============================================================================


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


@dataclass(frozen=True)
class CaseData:
    series_id: str
    image_itk: sitk.Image
    image_array: np.ndarray
    gt_volume: np.ndarray
    mask_path: Path


@dataclass(frozen=True)
class SliceSample:
    series_id: str
    z: int


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
        raise ValueError(f"{annotations_csv} must contain column 'seriesuid'")
    if not {"SeriesID", "CID"}.issubset(df_links.columns):
        raise ValueError(f"{links_csv} must contain columns 'SeriesID' and 'CID'")

    mask_id_to_file: Dict[int, str] = {}
    for f in masks_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if "mask" in name and "contour" in name and "circle" not in name and "nodule" in name:
            try:
                mask_id_to_file[int(name.split("_")[0])] = name
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
            raise ValueError("--dataset-fraction must be in (0, 1]")
        n = max(1, int(math.ceil(len(volume_ids) * args.dataset_fraction)))
        volume_ids = volume_ids[:n]

    if args.max_cases is not None:
        volume_ids = volume_ids[: max(0, args.max_cases)]

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


def load_case(index: DatasetIndex, series_id: str) -> Optional[CaseData]:
    image_path = index.volumes_dir / f"{series_id}.mhd"
    if not image_path.is_file():
        return None

    links = index.df_links[index.df_links["SeriesID"].astype(str) == str(series_id)]
    if len(links) == 0:
        return None

    mask_id = int(links["CID"].iloc[0])
    mask_fname = index.mask_id_to_file.get(mask_id)
    if mask_fname is None:
        return None

    mask_path = index.masks_dir / mask_fname
    if not mask_path.is_file():
        return None

    image_itk = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    mask_itk = sitk.ReadImage(str(mask_path))
    gt_volume = (sitk.GetArrayFromImage(mask_itk) >= 0.5).astype(np.uint8)
    if image_array.shape != gt_volume.shape:
        raise ValueError(f"Shape mismatch for {series_id}: image {image_array.shape}, mask {gt_volume.shape}")
    return CaseData(series_id, image_itk, image_array, gt_volume, mask_path)


def positive_slices_for_case(case: CaseData, min_component_area: int = 1) -> List[int]:
    sums = case.gt_volume.reshape(case.gt_volume.shape[0], -1).sum(axis=1)
    return [int(z) for z in np.where(sums >= min_component_area)[0].tolist()]


def filter_positive_volume_ids(index: DatasetIndex, volume_ids: Sequence[str], min_component_area: int) -> List[str]:
    keep: List[str] = []
    print(f"Filtering {len(volume_ids)} volumes to those with GT-positive nodule slices...")
    for sid in tqdm(volume_ids, desc="positive-volume-filter"):
        case = load_case(index, sid)
        if case is None:
            continue
        if len(positive_slices_for_case(case, min_component_area=min_component_area)) > 0:
            keep.append(sid)
    return keep


# =============================================================================
# Split utilities
# =============================================================================


def read_case_list_file(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    ids: List[str] = []
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(".mhd"):
            line = line[:-4]
        ids.append(line)
    return ids


def write_case_list_file(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in ids) + ("\n" if ids else ""))


def split_volume_ids(
    volume_ids: Sequence[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    train_case_list: Optional[str] = None,
    val_case_list: Optional[str] = None,
    test_case_list: Optional[str] = None,
    shuffle_splits: bool = True,
) -> Dict[str, List[str]]:
    available = [str(x) for x in volume_ids]
    available_set = set(available)
    explicit_train = read_case_list_file(train_case_list)
    explicit_val = read_case_list_file(val_case_list)
    explicit_test = read_case_list_file(test_case_list)

    if explicit_train is not None or explicit_val is not None or explicit_test is not None:
        val_ids = [x for x in (explicit_val or []) if x in available_set]
        test_ids = [x for x in (explicit_test or []) if x in available_set]
        if explicit_train is None:
            used_nontrain = set(val_ids) | set(test_ids)
            train_ids = [x for x in available if x not in used_nontrain]
        else:
            train_ids = [x for x in explicit_train if x in available_set]
        overlap = (set(train_ids) & set(val_ids)) | (set(train_ids) & set(test_ids)) | (set(val_ids) & set(test_ids))
        if overlap:
            raise ValueError(f"Explicit split files overlap for {len(overlap)} SeriesUIDs, e.g. {sorted(overlap)[:5]}")
        used = set(train_ids) | set(val_ids) | set(test_ids)
        return {"train": train_ids, "val": val_ids, "test": test_ids, "all": [x for x in available if x in used]}

    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1")

    ids = list(available)
    if shuffle_splits:
        rng = np.random.default_rng(seed)
        ids = [str(x) for x in rng.permutation(ids)]

    n_total = len(ids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    if n_total > 1 and val_ratio > 0 and n_val == 0:
        n_val = 1
    if n_total > 2 and test_ratio > 0 and n_test == 0:
        n_test = 1
    if n_val + n_test >= n_total and n_total > 0:
        excess = n_val + n_test - (n_total - 1)
        reduce_test = min(excess, n_test)
        n_test -= reduce_test
        excess -= reduce_test
        n_val = max(0, n_val - excess)

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids, "all": ids}


def write_split_files(splits: Dict[str, List[str]], out_dir: Path) -> None:
    split_dir = out_dir / "splits"
    for name in ["train", "val", "test", "all"]:
        write_case_list_file(split_dir / f"{name}.txt", splits.get(name, []))


# =============================================================================
# Dataset
# =============================================================================


class LUNAPositiveSliceSegDataset(Dataset):
    def __init__(
        self,
        index: DatasetIndex,
        volume_ids: Sequence[str],
        use_triplet_channels: bool,
        use_ct_window: bool,
        hu_min: float,
        hu_max: float,
        min_component_area: int,
        cache_cases: bool = False,
        image_size: Optional[int] = None,
        augment_params: Optional[Dict] = None,
    ):
        self.index = index
        self.volume_ids = list(volume_ids)
        self.use_triplet_channels = use_triplet_channels
        self.use_ct_window = use_ct_window
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.min_component_area = min_component_area
        self.cache_cases = cache_cases
        self.image_size = image_size
        self.augment_params = augment_params or {"enabled": False}
        self._case_cache: Dict[str, CaseData] = {}
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError("No GT-positive slice samples were found for this split")

    def _load_case(self, series_id: str) -> CaseData:
        if self.cache_cases and series_id in self._case_cache:
            return self._case_cache[series_id]
        case = load_case(self.index, series_id)
        if case is None:
            raise FileNotFoundError(f"Could not load case {series_id}")
        if self.cache_cases:
            self._case_cache[series_id] = case
        return case

    def _build_samples(self) -> List[SliceSample]:
        samples: List[SliceSample] = []
        print(f"Building positive-slice dataset from {len(self.volume_ids)} volumes...")
        for sid in tqdm(self.volume_ids, desc="index-positive-slices"):
            case = load_case(self.index, sid)
            if case is None:
                continue
            for z in positive_slices_for_case(case, min_component_area=self.min_component_area):
                samples.append(SliceSample(series_id=sid, z=int(z)))
                if self.cache_cases:
                    self._case_cache[sid] = case
        print(f"Built {len(samples)} positive slice samples from {len(set(s.series_id for s in samples))} volumes")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        case = self._load_case(s.series_id)
        rgb = build_sam_input_slice(
            case.image_array,
            s.z,
            use_triplet_channels=self.use_triplet_channels,
            use_ct_window=self.use_ct_window,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
        )
        mask = case.gt_volume[s.z].astype(np.uint8)
        rgb, mask = resize_rgb_and_mask(rgb, mask, self.image_size)
        rgb, mask = apply_train_augmentations(rgb, mask, self.augment_params)
        image = numpy_rgb_to_sam_tensor(rgb)
        mask_t = torch.from_numpy(mask.astype(np.float32))[None]
        H, W = mask.shape
        return {
            "image": image,
            "mask": mask_t,
            "series_id": s.series_id,
            "z": int(s.z),
            "image_hw": torch.tensor([H, W], dtype=torch.long),
        }

def collate_seg(batch: List[Dict]) -> Dict:
    """Collate function shared by train/val/test slice datasets.

    Some datasets include image_hw explicitly, while case-level evaluation
    datasets may only provide image/mask/series_id/z. Build image_hw from the
    mask shape as a fallback so the same collate function is safe in both paths.
    """
    if "image_hw" in batch[0]:
        image_hw = torch.stack([b["image_hw"] for b in batch], dim=0)
    else:
        image_hw = torch.tensor(
            [[int(b["mask"].shape[-2]), int(b["mask"].shape[-1])] for b in batch],
            dtype=torch.long,
        )

    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "series_id": [b["series_id"] for b in batch],
        "z": torch.tensor([b["z"] for b in batch], dtype=torch.long),
        "image_hw": image_hw,
    }


# =============================================================================
# Model
# =============================================================================


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleDecoder(nn.Module):
    """Original lightweight decoder: refine final SAM image embedding, then predict."""

    def __init__(self, in_ch: int = 256, decoder_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, decoder_dim, dropout=dropout),
            ConvGNAct(decoder_dim, decoder_dim, dropout=dropout),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class DeepDecoder(nn.Module):
    """Deeper low-resolution decoder using only the final SAM image embedding."""

    def __init__(self, in_ch: int = 256, decoder_dim: int = 256, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        depth = max(1, int(depth))
        layers: List[nn.Module] = [ConvGNAct(in_ch, decoder_dim, dropout=dropout)]
        for _ in range(depth):
            layers.append(ConvGNAct(decoder_dim, decoder_dim, dropout=dropout))
        layers.append(nn.Conv2d(decoder_dim, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class ProgressiveUpsampleDecoder(nn.Module):
    """U-Net/FPN-style progressive upsampling decoder for original SAM.

    Original SAM's public image encoder returns a final dense embedding rather than
    the multi-scale feature pyramid returned by SAM2. Therefore, for original SAM,
    --decoder-type fpn means a top-down/progressive upsampling decoder starting
    from the final SAM embedding. It does not use true encoder skip connections,
    but it gives the decoder several higher-resolution refinement stages before
    the final mask prediction.
    """

    def __init__(self, in_ch: int = 256, decoder_dim: int = 256, num_levels: int = 3, smooth_blocks: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_levels = max(1, int(num_levels))
        smooth_blocks = max(1, int(smooth_blocks))
        self.proj = ConvGNAct(in_ch, decoder_dim, dropout=dropout)
        self.level_blocks = nn.ModuleList()
        for _ in range(self.num_levels):
            blocks: List[nn.Module] = []
            for _ in range(smooth_blocks):
                blocks.append(ConvGNAct(decoder_dim, decoder_dim, dropout=dropout))
            self.level_blocks.append(nn.Sequential(*blocks))
        self.out_conv = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(feat)
        for i, block in enumerate(self.level_blocks):
            if i > 0:
                x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(x)
        return self.out_conv(x)


class SAMEncoderSegmentationModel(nn.Module):
    """Original SAM image encoder + selectable supervised segmentation decoder."""

    def __init__(
        self,
        sam_model: nn.Module,
        head_dim: int = 256,
        freeze_encoder: bool = True,
        sam_feature_channels: int = 256,
        decoder_type: str = "simple",
        decoder_depth: int = 4,
        fpn_levels: int = 3,
        decoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.sam = sam_model
        self.image_encoder = sam_model.image_encoder
        self.freeze_encoder = bool(freeze_encoder)
        self.decoder_type = str(decoder_type)
        self.image_size = int(getattr(self.image_encoder, "img_size", 1024))

        # Only the image encoder is used. Keep prompt/mask decoders frozen because they are unused.
        for p in self.sam.parameters():
            p.requires_grad = False
        if not self.freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = True

        # Register SAM pixel statistics as buffers. They are in 0-255 RGB space.
        pixel_mean = getattr(sam_model, "pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1))
        pixel_std = getattr(sam_model, "pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1))
        self.register_buffer("pixel_mean", pixel_mean.detach().clone().float().view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", pixel_std.detach().clone().float().view(1, 3, 1, 1), persistent=False)

        if self.decoder_type == "simple":
            self.seg_head = SimpleDecoder(in_ch=sam_feature_channels, decoder_dim=head_dim, dropout=decoder_dropout)
        elif self.decoder_type == "deep":
            self.seg_head = DeepDecoder(in_ch=sam_feature_channels, decoder_dim=head_dim, depth=decoder_depth, dropout=decoder_dropout)
        elif self.decoder_type == "fpn":
            self.seg_head = ProgressiveUpsampleDecoder(
                in_ch=sam_feature_channels,
                decoder_dim=head_dim,
                num_levels=fpn_levels,
                smooth_blocks=decoder_depth,
                dropout=decoder_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder_type={decoder_type!r}. Use simple, deep, or fpn.")

    def preprocess_for_sam_encoder(self, x: torch.Tensor) -> torch.Tensor:
        # x is B,3,H,W in [0,255]. It is resized to SAM's fixed encoder size.
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.pixel_mean.to(dtype=x.dtype)) / self.pixel_std.to(dtype=x.dtype)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x_sam = self.preprocess_for_sam_encoder(x)
        if self.freeze_encoder:
            with torch.no_grad():
                feat = self.image_encoder(x_sam)
        else:
            feat = self.image_encoder(x_sam)
        return feat

    def forward(self, x: torch.Tensor, output_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if output_hw is None:
            output_hw = tuple(x.shape[-2:])
        feat = self.extract_features(x)
        logits_low = self.seg_head(feat)
        logits = F.interpolate(logits_low, size=output_hw, mode="bilinear", align_corners=False)
        return logits


def build_sam_model(args: argparse.Namespace, device: torch.device):
    if args.model_type not in sam_model_registry:
        raise ValueError(f"Unknown --model-type {args.model_type!r}. Available: {sorted(sam_model_registry.keys())}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    return sam


def build_seg_model(args: argparse.Namespace, device: torch.device) -> SAMEncoderSegmentationModel:
    sam = build_sam_model(args, device)
    decoder_dim = args.decoder_dim if args.decoder_dim is not None else args.head_dim
    model = SAMEncoderSegmentationModel(
        sam_model=sam,
        head_dim=decoder_dim,
        freeze_encoder=not args.unfreeze_encoder,
        sam_feature_channels=args.sam_feature_channels,
        decoder_type=args.decoder_type,
        decoder_depth=args.decoder_depth,
        fpn_levels=args.fpn_levels,
        decoder_dropout=args.decoder_dropout,
    ).to(device)
    return model

# =============================================================================
# Losses
# =============================================================================


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float, dice_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_with_logits(logits, targets)
    loss = bce_weight * bce + dice_weight * dloss
    return loss, {"loss": float(loss.detach()), "bce_loss": float(bce.detach()), "dice_loss": float(dloss.detach())}


def batch_dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy().astype(np.uint8)
    gt = (targets.detach().cpu().numpy() >= 0.5).astype(np.uint8)
    dices = []
    ious = []
    for i in range(pred.shape[0]):
        dices.append(dice_score(gt[i, 0], pred[i, 0]))
        ious.append(iou_score(gt[i, 0], pred[i, 0]))
    return {"Dice": float(np.mean(dices)) if dices else np.nan, "IoU": float(np.mean(ious)) if ious else np.nan}


# =============================================================================
# Output/experiment helpers
# =============================================================================


def build_experiment_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        stamp,
        args.run_name,
        "sam-posslice-seg",
        args.model_type,
        f"dec-{args.decoder_type}",
        f"ddim{args.decoder_dim if args.decoder_dim is not None else args.head_dim}",
        "aug" if args.augment else "noaug",
        f"sz{args.image_size}" if args.image_size else "native",
        "triplet" if args.use_triplet_channels else "singlech",
        "window" if args.use_ct_window else "normslice",
        f"ep{args.epochs}",
        f"bs{args.batch_size}",
        f"lr{str(args.lr).replace('.', 'p')}",
        f"val{str(args.val_ratio).replace('.', 'p')}",
        f"test{str(args.test_ratio).replace('.', 'p')}",
    ]
    if args.max_cases is not None:
        parts.append(f"max{args.max_cases}")
    if args.unfreeze_encoder:
        parts.append("unfreeze-encoder")
    return slugify("_".join(parts))


def setup_experiment_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.create_experiment_dir:
        name = args.experiment_name or build_experiment_name(args)
        out_dir = root / name
    else:
        out_dir = root
    out_dir.mkdir(parents=True, exist_ok=args.overwrite_experiment)
    return out_dir


def save_experiment_config(args: argparse.Namespace, out_dir: Path, device: torch.device, index: DatasetIndex, splits: Dict[str, List[str]]) -> None:
    payload = {
        "experiment": {
            "experiment_dir": str(out_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "command": " ".join(os.sys.argv),
        },
        "runtime": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "arguments": namespace_to_plain_dict(args),
        "dataset": {
            "dataset_dir": str(index.dataset_dir),
            "volumes_dir": str(index.volumes_dir),
            "masks_dir": str(index.masks_dir),
            "annotations_csv": str(index.annotations_csv),
            "links_csv": str(index.links_csv),
            "n_selected_positive_volumes": len(index.volume_ids),
        },
        "splits": {k: {"n": len(v), "ids": list(v)} for k, v in splits.items()},
    }
    write_config(out_dir / "config.yaml", payload)


def plot_training_metrics(metrics_csv: Path, out_dir: Path) -> None:
    if not metrics_csv.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: could not import matplotlib for plots: {exc}")
        return
    df = pd.read_csv(metrics_csv)
    if df.empty or "epoch" not in df.columns:
        return

    def _plot(cols: List[str], name: str, title: str) -> None:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return
        plt.figure(figsize=(10, 6))
        for c in cols:
            plt.plot(df["epoch"], df[c], marker="o", linewidth=1.5, label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=200)
        plt.close()

    _plot([c for c in df.columns if "loss" in c.lower()], "training_validation_losses.png", "Training/validation losses")
    _plot([c for c in df.columns if c.endswith("Dice") or c.endswith("IoU")], "training_validation_metrics.png", "Training/validation metrics")


# =============================================================================
# Training
# =============================================================================


def run_epoch(
    model: SAMEncoderSegmentationModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp_dtype: str,
    bce_weight: float,
    dice_weight: float,
    grad_clip_norm: float,
    threshold: float,
    desc: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    # Keep the SAM model decoders frozen/unused; the encoder train state is controlled by model.train().
    totals = {"loss": 0.0, "bce_loss": 0.0, "dice_loss": 0.0, "Dice": 0.0, "IoU": 0.0}
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        H, W = int(masks.shape[-2]), int(masks.shape[-1])
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train), safe_autocast(device, amp_dtype):
            logits = model(images, output_hw=(H, W))
            loss, logs = segmentation_loss(logits, masks, bce_weight=bce_weight, dice_weight=dice_weight)
        if is_train:
            if scaler is not None and amp_dtype == "fp16":
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                optimizer.step()
        mlogs = batch_dice_iou_from_logits(logits, masks, threshold=threshold)
        bs = images.shape[0]
        for k in ["loss", "bce_loss", "dice_loss"]:
            totals[k] += logs[k] * bs
        totals["Dice"] += mlogs["Dice"] * bs
        totals["IoU"] += mlogs["IoU"] * bs
        n += bs
        pbar.set_postfix({k: f"{totals[k] / max(n, 1):.4f}" for k in ["loss", "Dice", "IoU"]})
    return {k: totals[k] / max(n, 1) for k in totals}


def save_checkpoint(
    path: Path,
    model: SAMEncoderSegmentationModel,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state": model.state_dict(),
        "args": namespace_to_plain_dict(args),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(model: SAMEncoderSegmentationModel, ckpt_path: Path, device: torch.device) -> Dict:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"WARNING missing keys while loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"WARNING unexpected keys while loading checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    return ckpt


# =============================================================================
# Evaluation/prediction
# =============================================================================


def extract_gt_nodules_3d(gt_volume: np.ndarray, spacing_zyx: Tuple[float, float, float], min_area: int = 1) -> List[Dict]:
    labeled, num = ndimage.label(gt_volume.astype(bool), structure=ndimage.generate_binary_structure(3, 2))
    nodules: List[Dict] = []
    voxel_volume = float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2])
    for k in range(1, num + 1):
        comp = labeled == k
        vox = int(comp.sum())
        if vox < min_area:
            continue
        coords = np.argwhere(comp)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        extent_vox = np.array([zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1], dtype=np.float32)
        extent_mm = extent_vox * np.array(spacing_zyx, dtype=np.float32)
        eq_diam_mm = float((6.0 * vox * voxel_volume / np.pi) ** (1.0 / 3.0))
        nodules.append({
            "nodule_id": int(k),
            "mask": comp.astype(np.uint8),
            "bbox_zyx": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
            "gt_volume_voxels": vox,
            "gt_volume_mm3": float(vox * voxel_volume),
            "gt_diameter_vox": float(max(extent_vox)),
            "gt_diameter_mm": float(max(extent_mm)),
            "gt_equivalent_diameter_mm": eq_diam_mm,
        })
    return nodules


def summarize_by_patient(df: pd.DataFrame, id_col: str, metric_cols: Sequence[str], count_name: str) -> pd.DataFrame:
    rows: List[Dict] = []
    if df.empty:
        return pd.DataFrame(rows)
    for sid, g in df.groupby(id_col):
        row = {"SeriesUID": sid, count_name: int(len(g))}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce") if c in g.columns else pd.Series([], dtype=float)
            row[f"{c}_mean"] = float(vals.mean()) if len(vals.dropna()) else np.nan
            row[f"{c}_median"] = float(vals.median()) if len(vals.dropna()) else np.nan
            row[f"{c}_std"] = float(vals.std()) if len(vals.dropna()) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


class CasePositiveSliceDataset(Dataset):
    def __init__(
        self,
        case: CaseData,
        use_triplet_channels: bool,
        use_ct_window: bool,
        hu_min: float,
        hu_max: float,
        min_component_area: int,
        image_size: Optional[int] = None,
    ):
        self.case = case
        self.use_triplet_channels = use_triplet_channels
        self.use_ct_window = use_ct_window
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.image_size = image_size
        self.zs = positive_slices_for_case(case, min_component_area=min_component_area)

    def __len__(self) -> int:
        return len(self.zs)

    def __getitem__(self, i: int) -> Dict:
        z = self.zs[i]
        rgb = build_sam_input_slice(
            self.case.image_array,
            z,
            use_triplet_channels=self.use_triplet_channels,
            use_ct_window=self.use_ct_window,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
        )
        # Use the same optional image resize at evaluation as during training.
        # Keep the target mask at native resolution; the model forward call passes
        # output_hw=(native_H, native_W), so predictions are written back correctly.
        rgb = resize_rgb_only(rgb, self.image_size)
        H, W = self.case.gt_volume.shape[1:]
        return {
            "image": numpy_rgb_to_sam_tensor(rgb),
            "mask": torch.from_numpy(self.case.gt_volume[z].astype(np.float32))[None],
            "series_id": self.case.series_id,
            "z": int(z),
            "image_hw": torch.tensor([H, W], dtype=torch.long),
        }

def predict_case_positive_slices(
    model: SAMEncoderSegmentationModel,
    case: CaseData,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, List[Dict]]:
    ds = CasePositiveSliceDataset(
        case,
        use_triplet_channels=args.use_triplet_channels,
        use_ct_window=args.use_ct_window,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        min_component_area=args.min_component_area,
        image_size=args.image_size,
    )
    pred_volume = np.zeros_like(case.gt_volume, dtype=np.uint8)
    slice_rows: List[Dict] = []
    if len(ds) == 0:
        return pred_volume, slice_rows
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_seg,
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            H, W = int(case.gt_volume.shape[1]), int(case.gt_volume.shape[2])
            with safe_autocast(device, args.amp_dtype):
                logits = model(images, output_hw=(H, W))
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            gt_np = masks.detach().cpu().numpy()
            for i, z in enumerate(batch["z"].tolist()):
                prob = probs[i, 0]
                pred = (prob >= args.threshold).astype(np.uint8)
                gt = (gt_np[i, 0] >= 0.5).astype(np.uint8)
                pred_volume[int(z)] = pred
                m = segmentation_metrics(gt, pred, spacing_zyx=None)
                slice_rows.append({
                    "SeriesUID": case.series_id,
                    "z": int(z),
                    "prob_mean": float(prob.mean()),
                    "prob_max": float(prob.max()),
                    **m,
                })
    return pred_volume, slice_rows


def evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    tag: str,
    index: DatasetIndex,
    test_ids: Sequence[str],
    out_dir: Path,
    device: torch.device,
) -> None:
    eval_dir = out_dir / "test_predictions" / tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    model = build_seg_model(args, device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    slice_rows: List[Dict] = []
    patient_volume_rows: List[Dict] = []
    nodule_rows: List[Dict] = []
    error_rows: List[Dict] = []
    pred_root = eval_dir / "predicted_volumes"

    for sid in tqdm(test_ids, desc=f"test-{tag}"):
        try:
            case = load_case(index, sid)
            if case is None:
                error_rows.append({"SeriesUID": sid, "status": "missing"})
                continue
            pred_volume, rows = predict_case_positive_slices(model, case, args, device)
            slice_rows.extend(rows)
            spacing = get_spacing_zyx(case.image_itk)

            # Full volume metrics, with predictions zero outside GT-positive slices.
            vol_metrics = segmentation_metrics(case.gt_volume, pred_volume, spacing_zyx=spacing)
            patient_volume_rows.append({
                "SeriesUID": sid,
                "status": "ok",
                "n_positive_slices": int(len(rows)),
                "n_slices": int(case.gt_volume.shape[0]),
                "height": int(case.gt_volume.shape[1]),
                "width": int(case.gt_volume.shape[2]),
                **vol_metrics,
            })

            # Per-nodule 3D metrics. If a patient has >1 nodule, patient_nodule_summary.csv gives mean/median.
            nodules = extract_gt_nodules_3d(case.gt_volume, spacing, min_area=args.min_component_area)
            for n in nodules:
                gt_n = n["mask"].astype(np.uint8)
                pred_n = (pred_volume.astype(bool) & gt_n.astype(bool)).astype(np.uint8) if args.nodule_metric_intersection_only else pred_volume.astype(np.uint8)
                nm = segmentation_metrics(gt_n, pred_n, spacing_zyx=spacing)
                nodule_rows.append({
                    "SeriesUID": sid,
                    "nodule_id": n["nodule_id"],
                    "bbox_zyx": str(n["bbox_zyx"]),
                    "GT_diameter_vox": n["gt_diameter_vox"],
                    "GT_diameter_mm": n["gt_diameter_mm"],
                    "GT_equivalent_diameter_mm": n["gt_equivalent_diameter_mm"],
                    "GT_volume_voxels": n["gt_volume_voxels"],
                    "GT_volume_mm3": n["gt_volume_mm3"],
                    **nm,
                })

            if args.save_volumes:
                write_pred_volume(pred_volume, case.image_itk, pred_root / f"{sid}_pred_{tag}.nii.gz")
                if args.save_gt_volume:
                    write_pred_volume(case.gt_volume, case.image_itk, pred_root / "gt" / f"{sid}_gt.nii.gz")
        except Exception as exc:
            if args.fail_fast:
                raise
            error_rows.append({"SeriesUID": sid, "status": "error", "error": repr(exc)})
            print(f"ERROR {sid}: {repr(exc)}")
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pd.DataFrame(slice_rows).to_csv(eval_dir / "slice_metrics.csv", index=False)
        pd.DataFrame(patient_volume_rows).to_csv(eval_dir / "patient_volume_metrics.csv", index=False)
        if nodule_rows:
            pd.DataFrame(nodule_rows).to_csv(eval_dir / "nodule_metrics.csv", index=False)

    slice_df = pd.DataFrame(slice_rows)
    vol_df = pd.DataFrame(patient_volume_rows)
    nod_df = pd.DataFrame(nodule_rows)
    err_df = pd.DataFrame(error_rows)

    slice_df.to_csv(eval_dir / "slice_metrics.csv", index=False)
    vol_df.to_csv(eval_dir / "patient_volume_metrics.csv", index=False)
    if not err_df.empty:
        err_df.to_csv(eval_dir / "errors.csv", index=False)

    slice_summary = summarize_by_patient(
        slice_df,
        id_col="SeriesUID",
        metric_cols=["Dice", "IoU", "precision", "recall", "f1", "volume_similarity", "prob_mean", "prob_max"],
        count_name="n_positive_slices",
    )
    slice_summary.to_csv(eval_dir / "patient_slice_summary.csv", index=False)

    if not nod_df.empty:
        nod_df.to_csv(eval_dir / "nodule_metrics.csv", index=False)
        nod_summary = summarize_by_patient(
            nod_df,
            id_col="SeriesUID",
            metric_cols=["Dice", "IoU", "precision", "recall", "f1", "HD95_mm", "ASSD_mm", "volume_similarity"],
            count_name="n_nodules",
        )
        nod_summary.to_csv(eval_dir / "patient_nodule_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(eval_dir / "nodule_metrics.csv", index=False)
        pd.DataFrame().to_csv(eval_dir / "patient_nodule_summary.csv", index=False)

    summary = {
        "tag": tag,
        "checkpoint": str(checkpoint_path),
        "n_test_requested": len(test_ids),
        "n_test_ok": int((vol_df["status"] == "ok").sum()) if "status" in vol_df.columns else int(len(vol_df)),
        "n_errors": int(len(error_rows)),
    }
    for prefix, df in [("slice", slice_df), ("volume", vol_df), ("nodule", nod_df)]:
        if df.empty:
            continue
        for c in ["Dice", "IoU", "precision", "recall", "f1", "HD95_mm", "ASSD_mm", "volume_similarity"]:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                summary[f"{prefix}_{c}_mean"] = float(vals.mean()) if len(vals.dropna()) else np.nan
                summary[f"{prefix}_{c}_median"] = float(vals.median()) if len(vals.dropna()) else np.nan
    pd.DataFrame([summary]).to_csv(eval_dir / "summary.csv", index=False)
    write_json(eval_dir / "summary.json", summary)
    print(f"Saved test outputs for {tag}: {eval_dir}")
    if "volume_Dice_mean" in summary:
        print(f"{tag} mean volume Dice: {summary['volume_Dice_mean']:.6f}")


def run_test_predictions_after_training(args: argparse.Namespace, out_dir: Path, index: DatasetIndex, splits: Dict[str, List[str]], device: torch.device) -> None:
    if not args.run_test_after_training:
        print("Skipping post-training test prediction because --no-run-test-after-training was set.")
        return
    test_ids = splits.get("test", [])
    if not test_ids:
        print("Skipping post-training test prediction because test split is empty.")
        return
    runs = [("best_model", out_dir / "best_model.pt"), ("last_model", out_dir / "last_model.pt")]
    for tag, ckpt in runs:
        if not ckpt.exists():
            print(f"WARNING: missing checkpoint for {tag}: {ckpt}")
            continue
        evaluate_checkpoint(args, ckpt, tag, index, test_ids, out_dir, device)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    out_dir = setup_experiment_dir(args)
    print(f"Using device: {device}")
    print(f"Experiment directory: {out_dir}")

    index = build_dataset_index(args)
    positive_ids = filter_positive_volume_ids(index, index.volume_ids, min_component_area=args.min_component_area)
    index = DatasetIndex(
        dataset_dir=index.dataset_dir,
        volumes_dir=index.volumes_dir,
        masks_dir=index.masks_dir,
        annotations_csv=index.annotations_csv,
        links_csv=index.links_csv,
        df_annotations=index.df_annotations,
        df_links=index.df_links,
        mask_id_to_file=index.mask_id_to_file,
        volume_ids=positive_ids,
    )
    splits = split_volume_ids(
        index.volume_ids,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_case_list=args.train_case_list,
        val_case_list=args.val_case_list,
        test_case_list=args.test_case_list,
        shuffle_splits=args.shuffle_splits,
    )
    write_split_files(splits, out_dir)
    save_experiment_config(args, out_dir, device, index, splits)
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, all={len(splits['all'])}")
    print(f"Saved config: {out_dir / 'config.yaml'}")

    train_ds = LUNAPositiveSliceSegDataset(
        index,
        splits["train"],
        use_triplet_channels=args.use_triplet_channels,
        use_ct_window=args.use_ct_window,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        min_component_area=args.min_component_area,
        cache_cases=args.cache_cases,
        image_size=args.image_size,
        augment_params=build_augment_params(args),
    )
    val_ids = splits["val"] if splits["val"] else splits["train"]
    val_ds = LUNAPositiveSliceSegDataset(
        index,
        val_ids,
        use_triplet_channels=args.use_triplet_channels,
        use_ct_window=args.use_ct_window,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        min_component_area=args.min_component_area,
        cache_cases=args.cache_cases,
        image_size=args.image_size,
        augment_params={"enabled": False},
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_seg,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_seg,
        persistent_workers=args.num_workers > 0,
    )

    model = build_seg_model(args, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. This should not happen because the segmentation head should be trainable.")
    print(f"Decoder: type={args.decoder_type}, dim={args.decoder_dim if args.decoder_dim is not None else args.head_dim}, depth={args.decoder_depth}, fpn_levels={args.fpn_levels}, dropout={args.decoder_dropout}")
    print(f"Augmentation enabled: {args.augment}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp_dtype == "fp16"))

    best_val = float("inf")
    bad_epochs = 0
    rows: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        train_log = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_dtype=args.amp_dtype,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            grad_clip_norm=args.grad_clip_norm,
            threshold=args.threshold,
            desc=f"train {epoch}/{args.epochs}",
        )
        with torch.no_grad():
            val_log = run_epoch(
                model,
                val_loader,
                None,
                None,
                device,
                amp_dtype=args.amp_dtype,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                grad_clip_norm=args.grad_clip_norm,
                threshold=args.threshold,
                desc=f"val {epoch}/{args.epochs}",
            )
        scheduler.step()
        row = {
            "epoch": int(epoch),
            **{f"train_{k}": v for k, v in train_log.items()},
            **{f"val_{k}": v for k, v in val_log.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        write_json(out_dir / "metrics.json", rows)
        print(
            f"Epoch {epoch:03d}: train_loss={train_log['loss']:.5f}, val_loss={val_log['loss']:.5f}, "
            f"val_Dice={val_log['Dice']:.5f}, val_IoU={val_log['IoU']:.5f}, lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        save_checkpoint(out_dir / "last_model.pt", model, optimizer, epoch, best_val, args)
        if val_log["loss"] < best_val - args.min_delta:
            best_val = val_log["loss"]
            bad_epochs = 0
            save_checkpoint(out_dir / "best_model.pt", model, optimizer, epoch, best_val, args)
            print(f"  saved best_model.pt with val_loss={best_val:.5f}")
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping after {bad_epochs} bad epochs.")
                break

    plot_training_metrics(out_dir / "metrics.csv", out_dir)
    run_test_predictions_after_training(args, out_dir, index, splits, device)
    print(f"Done. Output: {out_dir}")


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train original-SAM encoder + supervised segmentation head on LUNA positive slices.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset paths
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--volumes-dir", default=None)
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--annotations-csv", default=None)
    parser.add_argument("--links-csv", default=None)
    parser.add_argument("--case-list", default=None, help="Optional global case filter, one SeriesUID per line.")
    parser.add_argument("--train-case-list", default=None)
    parser.add_argument("--val-case-list", default=None)
    parser.add_argument("--test-case-list", default=None)
    parser.add_argument("--only-annotated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-fraction", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle global case order before fraction/max filtering.")
    parser.add_argument("--shuffle-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)

    # Model
    parser.add_argument("--model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--checkpoint", required=True, help="Original SAM checkpoint, e.g. sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-feature-channels", type=int, default=256, help="SAM image encoder output channels. Original SAM ViT models normally use 256.")
    parser.add_argument("--head-dim", type=int, default=256, help="Backward-compatible decoder channel width. Used when --decoder-dim is not set.")
    parser.add_argument("--decoder-type", choices=["simple", "deep", "fpn"], default="simple", help="Segmentation decoder/head. simple keeps the original head, deep uses more conv blocks, fpn uses progressive U-Net/FPN-style upsampling from the final SAM embedding.")
    parser.add_argument("--decoder-dim", type=int, default=None, help="Decoder channel width. If omitted, --head-dim is used for backward compatibility.")
    parser.add_argument("--decoder-depth", type=int, default=4, help="For deep: number of ConvGNAct blocks. For fpn: smoothing ConvGNAct blocks per upsampling level.")
    parser.add_argument("--fpn-levels", type=int, default=3, help="For original SAM, number of progressive upsampling/refinement levels when --decoder-type fpn.")
    parser.add_argument("--decoder-dropout", type=float, default=0.0, help="Optional Dropout2d probability inside decoder blocks.")
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Fine-tune SAM image encoder as well as the segmentation head.")

    # Runtime/training
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--cache-cases", action="store_true")

    # Augmentation; applied only to the training split after optional resize.
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False, help="Enable conservative 2D augmentations for training slices only.")
    parser.add_argument("--aug-hflip-p", type=float, default=0.5)
    parser.add_argument("--aug-vflip-p", type=float, default=0.5)
    parser.add_argument("--aug-rotation-deg", type=float, default=15.0)
    parser.add_argument("--aug-shift-px", type=float, default=16.0)
    parser.add_argument("--aug-scale-min", type=float, default=0.90)
    parser.add_argument("--aug-scale-max", type=float, default=1.10)
    parser.add_argument("--aug-intensity-p", type=float, default=0.8)
    parser.add_argument("--aug-brightness", type=float, default=0.10, help="Brightness jitter as a fraction of 255.")
    parser.add_argument("--aug-contrast", type=float, default=0.10, help="Contrast jitter fraction around 1.0.")
    parser.add_argument("--aug-noise-std", type=float, default=0.02, help="Gaussian noise std as a fraction of 255.")
    parser.add_argument("--aug-blur-p", type=float, default=0.10, help="Probability of mild 3x3 Gaussian blur.")

    # CT/slice input
    parser.add_argument("--use-triplet-channels", action="store_true", help="Use z-1/z/z+1 as RGB channels.")
    parser.add_argument("--use-ct-window", action=argparse.BooleanOptionalAction, default=True, help="Use fixed CT window before uint8 conversion. If false, per-slice min-max normalization is used.")
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=400.0)
    parser.add_argument("--image-size", type=int, default=None, help="Optional square resize before SAM preprocessing. Training masks are resized to this size; evaluation predictions are written back at native size.")
    parser.add_argument("--min-component-area", type=int, default=1)

    # Outputs
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="luna_sam_train")
    parser.add_argument("--create-experiment-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--overwrite-experiment", action="store_true")
    parser.add_argument("--save-volumes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gt-volume", action="store_true")
    parser.add_argument("--run-test-after-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--nodule-metric-intersection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, per-nodule metrics use pred AND nodule_gt to avoid penalizing predictions belonging to other nodules. "
            "If false, each nodule GT is compared against the whole patient prediction volume."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not (0 <= args.val_ratio < 1):
        raise ValueError("--val-ratio must be in [0,1)")
    if not (0 <= args.test_ratio < 1):
        raise ValueError("--test-ratio must be in [0,1)")
    if args.val_ratio + args.test_ratio >= 1 and not (args.train_case_list or args.val_case_list or args.test_case_list):
        raise ValueError("Need --val-ratio + --test-ratio < 1 unless explicit split files are used")
    if args.bce_weight < 0 or args.dice_weight < 0 or (args.bce_weight + args.dice_weight) <= 0:
        raise ValueError("Need non-negative --bce-weight/--dice-weight with positive sum")
    if not (0 <= args.threshold <= 1):
        raise ValueError("--threshold must be in [0,1]")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive when provided")
    if args.decoder_dim is not None and args.decoder_dim <= 0:
        raise ValueError("--decoder-dim must be positive when provided")
    if args.decoder_depth <= 0:
        raise ValueError("--decoder-depth must be positive")
    if args.fpn_levels <= 0:
        raise ValueError("--fpn-levels must be positive")
    if not (0.0 <= args.decoder_dropout < 1.0):
        raise ValueError("--decoder-dropout must be in [0,1)")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()