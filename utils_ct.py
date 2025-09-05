# utils_ct.py
# Helper functions for the AI Stroke Triage demo
# Educational use only. Not for clinical use.

import io
from typing import Dict, Any

import numpy as np
import pydicom
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated images from exports

# ----------------- DICOM detection -----------------
def _is_dicom(buf: bytes) -> bool:
    """Quick check for DICOM magic ('DICM') at offset 0x80."""
    return len(buf) > 0x80 + 4 and buf[0x80:0x80 + 4] == b"DICM"


# ----------------- PIL loader (safer) -----------------
def _load_pil_gray(data: bytes) -> np.ndarray:
    """Open a common image (PNG/JPG) via PIL and return uint8 grayscale."""
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("L")
        return np.asarray(img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Unsupported or corrupted image file.") from e


# ----------------- Public API: load CT as grayscale -----------------
def load_image_any(file) -> np.ndarray:
    """
    Load DICOM or standard image into grayscale numpy (H x W) uint8, scaled 0..255.
    Supports .dcm, .png, .jpg, .jpeg

    Raises:
        ValueError: if the file is unsupported or corrupted.
    """
    name = getattr(file, "name", "")
    data = file.read() if hasattr(file, "read") else file
    try:
        if name.lower().endswith(".dcm") or _is_dicom(data):
            ds = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array.astype(np.float32)

            # Apply rescale slope & intercept if present
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

            # Normalize to 0..255 for display/net
            arr = arr - arr.min()
            arr = arr / (arr.max() + 1e-6)
            return (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            return _load_pil_gray(data)
    except Exception as e:
        raise ValueError(f"Unsupported or corrupted file: {name}") from e
    finally:
        if hasattr(file, "seek"):
            file.seek(0)


# ----------------- Windowing utilities -----------------
WINDOW_PRESETS = {
    "brain": (40, 80),      # WL=40, WW=80
    "subdural": (50, 130),
    "stroke": (35, 35),     # sometimes used for early ischemia
    "bone": (600, 2000),
}

def apply_window(img_gray: np.ndarray, level: float, width: float) -> np.ndarray:
    """Generic CT window; returns uint8 image."""
    img = img_gray.astype(np.float32)
    low, high = level - width / 2.0, level + width / 2.0
    img = (img - low) / max(high - low, 1e-6)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def apply_brain_window(img_gray: np.ndarray, level: int = 40, width: int = 80) -> np.ndarray:
    """Back-compat helper for brain window."""
    return apply_window(img_gray, level, width)

def apply_window_preset(img_gray: np.ndarray, name: str = "brain") -> np.ndarray:
    """Window using a named preset (see WINDOW_PRESETS)."""
    wl, ww = WINDOW_PRESETS.get(name.lower(), WINDOW_PRESETS["brain"])
    return apply_window(img_gray, wl, ww)


# ----------------- Blur/quality metric -----------------
def blur_score_variance_of_laplacian(img_gray: np.ndarray) -> float:
    """
    Simple blur metric; lower values ≈ more blur.
    Typical heuristic threshold ~60–120 for CT screenshots.
    """
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


# ----------------- Heatmap compositor -----------------
def compose_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4,
                    cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Blend CAM (HxW in [0,1]) onto RGB image; returns uint8 HxWx3.
    """
    cam = np.clip(cam, 0.0, 1.0).astype(np.float32)
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heat = (cam * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cmap)
    return cv2.addWeighted(rgb, 1.0 - alpha, heat, alpha, 0.0)


# ----------------- DICOM metadata (safe subset) -----------------
def dicom_meta_safe(ds) -> Dict[str, Any]:
    """
    Extract non-identifying technical DICOM tags useful for QA.
    """
    def get(tag, default=None):
        return getattr(ds, tag, default)

    meta = {
        "Modality": str(get("Modality", "")),
        "Manufacturer": str(get("Manufacturer", "")),
        "KVP": float(get("KVP", 0)) if hasattr(ds, "KVP") else None,
        "SliceThickness": float(get("SliceThickness", 0)) if hasattr(ds, "SliceThickness") else None,
        "PixelSpacing": [float(x) for x in get("PixelSpacing", [])] if hasattr(ds, "PixelSpacing") else None,
        "Rows": int(get("Rows", 0)) if hasattr(ds, "Rows") else None,
        "Columns": int(get("Columns", 0)) if hasattr(ds, "Columns") else None,
        "BitsStored": int(get("BitsStored", 0)) if hasattr(ds, "BitsStored") else None,
        "RescaleSlope": float(get("RescaleSlope", 1.0)) if hasattr(ds, "RescaleSlope") else 1.0,
        "RescaleIntercept": float(get("RescaleIntercept", 0.0)) if hasattr(ds, "RescaleIntercept") else 0.0,
    }
    return meta

def extract_dicom_meta(file) -> Dict[str, Any]:
    """
    If the uploaded file is DICOM, return a safe metadata dict; otherwise {}.
    """
    name = getattr(file, "name", "")
    data = file.read() if hasattr(file, "read") else file
    try:
        if name.lower().endswith(".dcm") or _is_dicom(data):
            ds = pydicom.dcmread(io.BytesIO(data))
            return dicom_meta_safe(ds)
        return {}
    except Exception:
        return {}
    finally:
        if hasattr(file, "seek"):
            file.seek(0)


# ----------------- Grad-CAM -----------------
def gradcam(model, x: torch.Tensor, target_class: int, layer_name: str = "layer4") -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a ResNet18-like model.
    Returns HxW heatmap normalized to [0,1].
    """
    feats = []
    grads = []

    def f_hook(_, __, output):
        feats.append(output)

    def b_hook(_, grad_in, grad_out):
        grads.append(grad_out[0])

    layer = dict(model.named_modules())[layer_name]
    h1 = layer.register_forward_hook(f_hook)
    h2 = layer.register_full_backward_hook(b_hook)

    model.zero_grad()
    logits = model(x)
    score = logits[0, target_class]
    score.backward()

    fmap = feats[0].detach()[0]
    grad = grads[0].detach()[0]
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * fmap).sum(0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    cam = cam.cpu().numpy()

    h1.remove()
    h2.remove()
    return cam
