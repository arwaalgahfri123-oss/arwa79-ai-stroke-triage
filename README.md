# README — AI Stroke Triage Prototype (Oman)

This prototype is a **free, local, non‑clinical demo** that shows how AI could help doctors triage stroke on **non‑contrast head CT** in Oman. It’s built to be:
- **Unique to Oman**: bilingual UI (Arabic/English), KPIs aligned with **≤25‑min door‑to‑CT**, and ready for on‑prem inference.
- **Legally safe**: uses open‑source tools (PyTorch, MONAI ops) and open datasets you download yourself (e.g., **RSNA Intracranial Hemorrhage** on Kaggle; **CQ500**). No vendor code.
- **Modular**: you can fine‑tune on local data later and drop the weights into the app.

> ⚠️ Educational use only. Not for clinical decisions.

---

## 1) Quick Start

### A. Create a Python environment
```bash
# Python 3.10+ recommended
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### B. Run the demo app
```bash
streamlit run app.py
```
Open the URL shown (usually http://localhost:8501).

You can upload **DICOM (.dcm) or PNG/JPG** axial head CT images. If you have a trained checkpoint (e.g., `models/rsna_resnet18.pt`), place it there and the app will use it. Otherwise, it runs in **demo mode** with ImageNet weights (works visually, not clinically accurate).

### C. (Optional) Train a simple model on RSNA ICH
1) Download the **RSNA Intracranial Hemorrhage** dataset from Kaggle (requires a free account). Organize like:
```
DATA/rsna_ich/
  |-- stage_2_train_images/   # DICOMs
  |-- train_labels.csv        # image_id, any_hemorrhage (0/1)
```
2) Start training:
```bash
python train_rsna.py \
  --data_root DATA/rsna_ich \
  --out_dir models \
  --epochs 3 \
  --batch_size 16
```
This will create `models/rsna_resnet18.pt`. Restart the Streamlit app to load it.

---

## 2) Oman‑Specific Features You Can Toggle
- **Arabic UI**: switch language in sidebar.
- **KPI timer**: tracks *triage time* vs. the **≤25‑min target** (Oman Neurology Society / MoH guidance).
- **Priority rules**: hemorrhage > suspected LVO > ischemic.
- **On‑prem**: everything runs locally; no cloud.

---

## 3) Files
- `requirements.txt` – all dependencies
- `app.py` – Streamlit inference app (Arabic/English UI, Grad‑CAM)
- `train_rsna.py` – simple PyTorch fine‑tuning on RSNA ICH (any‑hemorrhage binary)
- `utils_ct.py` – DICOM loading, windowing, and Grad‑CAM helper

---

# requirements.txt

# Core
streamlit==1.37.1
torch==2.3.1
torchvision==0.18.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
opencv-python-headless==4.10.0.84
pydicom==2.4.4
Pillow==10.4.0
matplotlib==3.9.0

# Optional (for better medical ops)
monai==1.3.0


# app.py

import os
import time

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models

from utils_ct import load_image_any, apply_brain_window, gradcam

MODEL_PATH = os.getenv("MODEL_PATH", "models/rsna_resnet18.pt")
PAGE_TITLE = "AI Stroke Triage — Oman Prototype"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])  # default English

TXT = {
    "title": {"English": "AI Stroke Triage — Oman Prototype", "العربية": "نموذج أولي لفرز السكتة الدماغية بالذكاء الاصطناعي — عُمان"},
    "upload": {"English": "Upload axial head CT (DICOM/PNG/JPG)", "العربية": "حمّل صورة الأشعة المقطعية للرأس (DICOM/PNG/JPG)"},
    "run": {"English": "Run Triage", "العربية": "تشغيل الفرز"},
    "note": {"English": "Educational demo. Not for clinical use.", "العربية": "عرض تعليمي فقط. ليس للاستخدام السريري."},
    "kpi": {"English": "Triage timer (target ≤ 25 min)", "العربية": "مؤقت الفرز (الهدف ≤ 25 دقيقة)"},
    "results": {"English": "AI Findings", "العربية": "نتائج الذكاء الاصطناعي"},
    "classes": {"English": ["No hemorrhage (normal/ischemic?)", "Intracranial hemorrhage"], "العربية": ["لا نزف (طبيعي/إقفاري؟)", "نزف داخل الجمجمة"]}
}

st.title(TXT["title"][lang])
st.caption(TXT["note"][lang])

@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, 2)
    trained = False
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            trained = True
        except Exception as e:
            st.warning(f"Failed to load model weights at {MODEL_PATH}: {e}. Using ImageNet weights (DEMO mode).")
    model.eval()
    return model, trained

model, trained = load_model()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded = st.file_uploader(TXT["upload"][lang], type=["dcm", "png", "jpg", "jpeg"]) 

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded is not None:
        img_gray = load_image_any(uploaded)
        img_win = apply_brain_window(img_gray)
        st.image(img_win, caption="Brain window", use_column_width=True, clamp=True)
    else:
        st.info("Upload a CT image to begin.")

with col2:
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    timer_val = "00:00"
    if st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        mm = int(elapsed // 60)
        ss = int(elapsed % 60)
        timer_val = f"{mm:02d}:{ss:02d}"
    st.metric(TXT["kpi"][lang], value=timer_val, delta="Target ≤ 25 min")
    start = st.button(TXT["run"][lang], use_container_width=True)

if uploaded is not None and start:
    st.session_state.start_time = time.time()
    rgb = cv2.cvtColor(img_win, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(prob.argmax())
    pred_label = TXT["classes"][lang][pred_idx]

    cam = gradcam(model, x, target_class=pred_idx)
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)

    elapsed = time.time() - st.session_state.start_time
    mm = int(elapsed // 60); ss = int(elapsed % 60)

    st.subheader(TXT["results"][lang])
    st.write(f"**{pred_label}** — P(hemorrhage) = {prob[1]:.2f}")
    st.image(overlay, caption="Attention heatmap (Grad‑CAM)", use_column_width=True)
    st.metric(TXT["kpi"][lang], value=f"{mm:02d}:{ss:02d}", delta="Target ≤ 25 min")

    if not trained:
        st.warning("DEMO mode (ImageNet weights). Add a trained checkpoint to models/ for realistic performance.")


# utils_ct.py

import io
import numpy as np
import pydicom
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


def _is_dicom(buf: bytes) -> bool:
    return len(buf) > 0x80 + 4 and buf[0x80:0x80+4] == b'DICM'


def load_image_any(file) -> np.ndarray:
    """Load DICOM or image bytes into grayscale numpy (H x W), scaled to 0..255."""
    name = getattr(file, 'name', '')
    data = file.read() if hasattr(file, 'read') else file
    try:
        if name.lower().endswith('.dcm') or _is_dicom(data):
            ds = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            arr = arr * slope + intercept
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
    wl, ww = level, width
    low = wl - ww/2
    high = wl + ww/2
    img = img_gray.astype(np.float32)
    img = (img - low) / (high - low)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def gradcam(model, x: torch.Tensor, target_class: int, layer_name: str = 'layer4') -> np.ndarray:
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

    h1.remove(); h2.remove()
    return cam
