# app.py
# Streamlit AI Stroke Triage — Oman Prototype
# Educational demo only. Not for clinical use.

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

from utils_ct import load_image_any, apply_brain_window, gradcam  # helper funcs

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/rsna_resnet18.pt")
PAGE_TITLE = "AI Stroke Triage — Oman Prototype"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# ---------- Language toggles ----------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])  # default English

TXT = {
    "title": {"English": "AI Stroke Triage — Oman Prototype",
              "العربية": "نموذج أولي لفرز السكتة الدماغية بالذكاء الاصطناعي — عُمان"},
    "upload": {"English": "Upload axial head CT (DICOM/PNG/JPG)",
               "العربية": "حمّل صورة الأشعة المقطعية للرأس (DICOM/PNG/JPG)"},
    "run": {"English": "Run Triage", "العربية": "تشغيل الفرز"},
    "note": {"English": "Educational demo. Not for clinical use.",
             "العربية": "عرض تعليمي فقط. ليس للاستخدام السريري."},
    "kpi": {"English": "Triage timer (target ≤ 25 min)",
            "العربية": "مؤقت الفرز (الهدف ≤ 25 دقيقة)"},
    "results": {"English": "AI Findings", "العربية": "نتائج الذكاء الاصطناعي"},
    "classes": {
        "English": ["No hemorrhage (normal/ischemic?)", "Intracranial hemorrhage"],
        "العربية": ["لا نزف (طبيعي/إقفاري؟)", "نزف داخل الجمجمة"],
    },
}

st.title(TXT["title"][lang])
st.caption(TXT["note"][lang])

# ---------- Model ----------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, 2)  # binary: no-ICH vs ICH

    trained = False
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            trained = True
        except Exception as e:
            st.warning(
                f"Failed to load model weights at {MODEL_PATH}: {e}. "
                "Using ImageNet weights (DEMO mode)."
            )
    model.eval()
    return model, trained

model, trained = load_model()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- UI ----------
uploaded = st.file_uploader(TXT["upload"][lang], type=["dcm", "png", "jpg", "jpeg"])

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded is not None:
        img_gray = load_image_any(uploaded)       # HxW uint8
        img_win = apply_brain_window(img_gray)    # windowed for viewing
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

    # Prepare tensor
    rgb = cv2.cvtColor(img_win, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(prob.argmax())
    pred_label = TXT["classes"][lang][pred_idx]

    # Grad-CAM heatmap
    cam = gradcam(model, x, target_class=pred_idx)        # HxW in [0,1]
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)

    elapsed = time.time() - st.session_state.start_time
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)

    st.subheader(TXT["results"][lang])
    st.write(f"**{pred_label}** — P(hemorrhage) = {prob[1]:.2f}")
    st.image(overlay, caption="Attention heatmap (Grad-CAM)", use_column_width=True)
    st.metric(TXT["kpi"][lang], value=f"{mm:02d}:{ss:02d}", delta="Target ≤ 25 min")

    if not trained:
        st.warning("DEMO mode (ImageNet weights). Add a trained checkpoint to models/ for realistic performance.")
