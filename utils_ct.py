# utils_ct.py
# Helper functions for the AI Stroke Triage demo
# Educational use only. Not for clinical use.

import io
import numpy as np
import pydicom
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


def _is_dicom(buf: bytes) -> bool:
    """Check if a file buffer is a DICOM by magic number."""
    return len(buf) > 0x80 + 4 and buf[0x80:0x80+4] == b'DICM'


def load_image_any(file) -> np.ndarray:
    """
    Load DICOM or image bytes into grayscale numpy (H x W), scaled 0..255.
    Supports .dcm, .png, .jpg, .jpeg
    """
    name = getattr(file, 'name', '')
    data = file.read() if hasattr(file, 'read') else file
    try:
        if name.lower().endswith('.dcm') or _is_dicom(data):
            ds = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array.astype(np.float32)

            # Apply rescale slope & intercept if present
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            arr = arr * slope + intercept

            # Normalize to 0..255
            arr = arr - arr.min()
            arr = arr / (arr.max() + 1e-6)
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            return arr
        else:
            img = Image.open(io.BytesIO(data)).convert('L')
            return np.array(img)
    finally:
        if hasattr(file, 'seek'):
            file.seek(0)


def apply_brain_window(img_gray: np.ndarray, level: int = 40, width: int = 80) -> np.ndarray:
    """
    Apply a simple brain window (WL=40, WW=80) to grayscale image.
    Returns 8-bit uint8 image.
    """
    wl, ww = level, width
    low = wl - ww/2
    high = wl + ww/2
    img = img_gray.astype(np.float32)
    img = (img - low) / (high - low)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def gradcam(model, x: torch.Tensor, target_class: int, layer_name: str = 'layer4') -> np.ndarray:
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
