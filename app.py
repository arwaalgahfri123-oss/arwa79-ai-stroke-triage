# app.py
# Streamlit AI Stroke Triage — Oman Prototype
# Educational demo only. Not for clinical use.

import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models

from utils_ct import load_image_any, apply_brain_window, gradcam  # helper funcs

# ---------- Assets ----------
LOGO = Path("logo.png")
ICON_TIMER = Path("icon_timer.png")
ICON_BRAIN = Path("brain_scan.png")
ICON_AI = Path("ai.png")

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/stroke_3class.pt")  # future 3-class checkpoint
PAGE_TITLE = "AI Stroke Triage — Oman Prototype"

st.set_page_config(page_title=PAGE_TITLE, page_icon="logo.png", layout="wide")

# ---------- Language toggle ----------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])

# 3-class labels (bilingual)
CLASS_NAMES = {
    "English": ["Normal / No acute stroke", "Ischemic stroke (suspected)", "Hemorrhagic stroke"],
    "العربية": ["طبيعي / لا سكتة حادة", "سكتة إقفارية (مشتبه بها)", "سكتة نزفية"],
}

TXT = {
    "title": {
        "English": "AI Stroke Triage — Oman Prototype",
        "العربية": "نموذج أولي لفرز السكتة الدماغية بالذكاء الاصطناعي — عُمان",
    },
    "upload": {
        "English": "Upload axial head CT (DICOM/PNG/JPG)",
        "العربية": "حمّل صورة الأشعة المقطعية للرأس (DICOM/PNG/JPG)",
    },
    "run": {"English": "Run Triage", "العربية": "تشغيل الفرز"},
    "note": {
        "English": "Educational demo. Not for clinical use.",
        "العربية": "عرض تعليمي فقط. ليس للاستخدام السريري.",
    },
    "kpi": {
        "English": "Triage timer (target ≤ 25 min)",
        "العربية": "مؤقت الفرز (الهدف ≤ 25 دقيقة)",
    },
    "results": {"English": "AI Findings", "العربية": "نتائج الذكاء الاصطناعي"},
    "classes": CLASS_NAMES,
}

# Bilingual developer credit
dev_text = (
    "Developed by <b>Arwa Alghafri</b>"
    if lang == "English"
    else "تم تطويره بواسطة <b>أروى الغافرية</b>"
)

# ---------- Header (logo + title + credit) ----------
col1, col2 = st.columns([1, 5])
with col1:
    if LOGO.exists():
        st.image(str(LOGO), width=80)
with col2:
    st.markdown(
        "<h2 style='margin-bottom:0; color:#3EC8E0;'>AI Stroke Triage — Oman</h2>"
        "<p style='opacity:.85; color:#EAEAEA;'>Bilingual demo • ≤25-min target • Vision 2040</p>"
        f"<p style='opacity:.7; font-size:0.9rem; color:#EAEAEA;'>{dev_text}</p>",
        unsafe_allow_html=True,
    )
st.caption(TXT["note"][lang])
st.divider()

# ---------- Model ----------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, 3)  # <<< three classes now

    trained = False
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
            # allow loading even if checkpoint head doesn't match exactly
            model.load_state_dict(state, strict=False)
            trained = True
        except Exception as e:
            st.warning(
                f"Failed to load model weights at {MODEL_PATH}: {e}. "
                "Using ImageNet weights (DEMO mode)."
            )
    model.eval()
    return model, trained

model, trained = load_model()

transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------- UI ----------
uploaded = st.file_uploader(TXT["upload"][lang], type=["dcm", "png", "jpg", "jpeg"])

left, right = st.columns([2, 1])

with left:
    if uploaded is not None:
        img_gray = load_image_any(uploaded)  # HxW uint8
        img_win = apply_brain_window(img_gray)  # windowed for viewing
        st.image(img_win, caption="Brain window", use_column_width=True, clamp=True)
    else:
        st.info("Upload a CT image to begin." if lang == "English" else "قم بتحميل صورة الأشعة المقطعية للبدء.")

with right:
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

    # Inference (3-class)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()  # shape (3,)

    pred_idx = int(prob.argmax())
    labels = CLASS_NAMES[lang]
    pred_label = labels[pred_idx]

    # Grad-CAM for the winning class
    cam = gradcam(model, x, target_class=pred_idx)  # HxW in [0,1]
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)

    elapsed = time.time() - st.session_state.start_time
    mm = int(elapsed // 60)
    ss = int(elapsed % 60)

    st.subheader(TXT["results"][lang])
    st.write(f"**{pred_label}**")

    # Per-class probabilities
    st.write(
        f"- {labels[0]}: **{prob[0]:.2f}**\n"
        f"- {labels[1]}: **{prob[1]:.2f}**\n"
        f"- {labels[2]}: **{prob[2]:.2f}**"
    )

    # Simple triage cue
    if prob[2] >= 0.60:
        st.error("High hemorrhage probability — prioritize urgent review."
                 if lang == "English"
                 else "احتمال مرتفع للنزف — أولوية قصوى للمراجعة.")
    elif prob[1] >= 0.60:
        st.warning("Possible ischemic stroke — consider urgent clinical correlation."
                   if lang == "English"
                   else "سكتة إقفارية محتملة — يُنصح بمراجعة سريرية عاجلة.")

    st.image(overlay, caption="Attention heatmap (Grad-CAM)", use_column_width=True)
    st.metric(TXT["kpi"][lang], value=f"{mm:02d}:{ss:02d}", delta="Target ≤ 25 min")

    if not trained:
        st.warning(
            "Running in DEMO mode (not trained for 3 classes yet). "
            "Add a 3-class checkpoint to models/ for realistic performance."
        )

# ---------- Optional: simple feature rows using your icons ----------
def feature_row(icon: Path, title_en: str, title_ar: str, desc_en: str, desc_ar: str):
    t = title_en if lang == "English" else title_ar
    d = desc_en if lang == "English" else desc_ar
    c1, c2 = st.columns([1, 12])
    with c1:
        if icon.exists():
            st.image(str(icon), width=40)
    with c2:
        st.markdown(f"**{t}** — {d}")

st.subheader("Key Features" if lang == "English" else "الميزات الأساسية")
feature_row(
    ICON_TIMER,
    "Faster triage", "فرز أسرع",
    "Meets ≤25-min door-to-CT KPI.", "يلبي مؤشر الأداء ≤ 25 دقيقة من الوصول إلى الأشعة."
)
feature_row(
    ICON_BRAIN,
    "CT-first workflow", "مسار عمل يعتمد على الأشعة المقطعية أولًا",
    "Designed for non-contrast head CT in Oman EDs.", "مصمم للأشعة المقطعية دون تباين في أقسام الطوارئ بعُمان."
)
feature_row(
    ICON_AI,
    "Decision support", "دعم القرار",
    "Grad-CAM shows where the model focuses.", "يوضح Grad-CAM مناطق اهتمام النموذج."
)

# ---------- Footer ----------
st.markdown("<hr style='opacity:.3;'>", unsafe_allow_html=True)
footer = (
    "© 2025 Developed by Arwa Alghafri — Educational Prototype"
    if lang == "English"
    else "© ٢٠٢٥ تم تطويره بواسطة أروى الغافرية — نموذج تعليمي"
)
st.caption(footer)
