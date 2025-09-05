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

from utils_ct import (
    load_image_any, apply_brain_window, apply_window_preset,
    blur_score_variance_of_laplacian, compose_heatmap,
    extract_dicom_meta, gradcam
)

# ---------- Assets ----------
LOGO = Path("logo.png")
ICON_TIMER = Path("icon_timer.png")
ICON_BRAIN = Path("brain_scan.png")
ICON_AI = Path("ai.png")

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/stroke_3class.pt")  # 3-class checkpoint when you train
PAGE_TITLE = "AI Stroke Triage — Oman Prototype"
st.set_page_config(page_title=PAGE_TITLE, page_icon="logo.png", layout="wide")

# ---------- Language toggle ----------
lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])

# Prob thresholds (sidebar sliders so they’re tunable & persistent)
st.sidebar.markdown("---")
st.sidebar.markdown("**Triage thresholds**" if lang == "English" else "**عتبات الفرز**")
HEMORRHAGE_THRESHOLD = st.sidebar.slider(
    "Hemorrhage alert" if lang == "English" else "تنبيه النزف",
    min_value=0.30, max_value=0.95, value=st.session_state.get("HEM_THR", 0.60), step=0.01
)
ISCHEMIC_THRESHOLD = st.sidebar.slider(
    "Ischemic alert" if lang == "English" else "تنبيه الإقفار",
    min_value=0.30, max_value=0.95, value=st.session_state.get("ISC_THR", 0.60), step=0.01
)
st.session_state["HEM_THR"] = HEMORRHAGE_THRESHOLD
st.session_state["ISC_THR"] = ISCHEMIC_THRESHOLD

# 3-class labels (bilingual)
CLASS_NAMES = {
    "English": ["Normal / No acute stroke", "Ischemic stroke (suspected)", "Hemorrhagic stroke"],
    "العربية": ["طبيعي / لا سكتة حادة", "سكتة إقفارية (مشتبه بها)", "سكتة نزفية"],
}

TXT = {
    "title": {"English": "AI Stroke Triage — Oman Prototype",
              "العربية": "نموذج أولي لفرز السكتة الدماغية بالذكاء الاصطناعي — عُمان"},
    "upload": {"English": "Upload axial head CT (DICOM/PNG/JPG)",
               "العربية": "حمّل صورة الأشعة المقطعية للرأس (DICOM/PNG/JPG)"},
    "run": {"English": "Run Triage", "العربية": "تشغيل الفرز"},
    "stop": {"English": "Stop", "العربية": "إيقاف"},
    "note": {"English": "Educational demo. Not for clinical use.",
             "العربية": "عرض تعليمي فقط. ليس للاستخدام السريري."},
    "kpi": {"English": "Triage timer (target ≤ 25 min)",
            "العربية": "مؤقت الفرز (الهدف ≤ 25 دقيقة)"},
    "results": {"English": "AI Findings", "العربية": "نتائج الذكاء الاصطناعي"},
}

# Bilingual developer credit
dev_text = ("Developed by <b>Arwa Alghafri</b>"
            if lang == "English"
            else "تم تطويره بواسطة <b>أروى الغافرية</b>")

# ---------- Header ----------
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

# ---------- Timer state & helpers ----------
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

def start_timer() -> None:
    st.session_state.start_time = time.time()
    st.session_state.timer_running = True

def stop_timer() -> None:
    st.session_state.timer_running = False

def read_timer() -> str:
    if st.session_state.timer_running and st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        mm = int(elapsed // 60); ss = int(elapsed % 60)
        return f"{mm:02d}:{ss:02d}"
    return "00:00"

# ---------- Model ----------
@st.cache_resource
def load_model(model_path: str) -> tuple[nn.Module, bool, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, 3)
    trained = False
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location="cpu")
            try:
                model.load_state_dict(state, strict=True)
                trained = True
            except RuntimeError:
                md = model.state_dict()
                filtered = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
                md.update(filtered)
                model.load_state_dict(md, strict=False)
                st.warning("Checkpoint partially loaded (shape mismatch). Running in partial/DEMO mode.")
                trained = len(filtered) > 0
        except Exception as e:
            st.warning(f"Failed to load weights at {model_path}: {e}. Using ImageNet weights (DEMO).")
    model.to(device).eval()
    return model, trained, device

model, trained, device = load_model(MODEL_PATH)
st.sidebar.caption(f"Compute device: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- File validation & pre-load image ----------
allowed_ext = {"dcm", "png", "jpg", "jpeg"}
uploaded = st.file_uploader(TXT["upload"][lang], type=list(allowed_ext))

img_win = None
is_dicom = False
meta = {}
if uploaded is not None:
    ext = uploaded.name.split(".")[-1].lower()
    is_dicom = (ext == "dcm")
    try:
        img_gray = load_image_any(uploaded)             # HxW uint8
        img_win = apply_brain_window(img_gray)          # default brain window
        # Optional: show subdural preset example (commented)
        # img_win = apply_window_preset(img_gray, "subdural")
        if is_dicom:
            meta = extract_dicom_meta(uploaded)
    except Exception:
        st.error("Could not read this file. Please upload a valid head CT DICOM/PNG/JPG."
                 if lang == "English" else "تعذر قراءة الملف. يرجى تحميل أشعة مقطعية للرأس بصيغة صحيحة.")
        img_win = None

# ---------- Layout ----------
start_clicked = False
left, right = st.columns([2, 1])

with left:
    if img_win is not None:
        st.image(img_win, caption="Brain window", use_container_width=True, clamp=True)

        # Blur/quality warning
        score = blur_score_variance_of_laplacian(img_win)
        if score < 80:
            st.warning("Image appears blurred (low sharpness)."
                       if lang == "English" else "تبدو الصورة غير واضحة (حدة منخفضة).")

        # Optional metadata for DICOMs
        if is_dicom and meta:
            with st.expander("DICOM technical info" if lang == "English" else "معلومات تقنية عن DICOM"):
                st.json(meta)
    else:
        st.info("Upload a CT image to begin." if lang == "English" else "قم بتحميل صورة الأشعة المقطعية للبدء.")

with right:
    if not trained:
        st.info("Running in DEMO mode (no 3-class checkpoint found). "
                "Add models/stroke_3class.pt for realistic performance.")
    st.metric(TXT["kpi"][lang], value=read_timer(), delta="Target ≤ 25 min")
    c1, c2 = st.columns(2)
    with c1:
        start_clicked = st.button(TXT["run"][lang], use_container_width=True)
    with c2:
        stop_clicked = st.button(TXT["stop"][lang], use_container_width=True)
    if start_clicked:
        start_timer()
    if stop_clicked:
        stop_timer()

# ---------- Inference ----------
if img_win is not None and start_clicked:
    try:
        rgb = cv2.cvtColor(img_win, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb)
        x = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (3,)

        pred_idx = int(prob.argmax())
        labels = CLASS_NAMES[lang]
        pred_label = labels[pred_idx]

        st.subheader(TXT["results"][lang])
        st.write(f"**{pred_label}**")
        st.write(f"- {labels[0]}: **{prob[0]:.2f}**")
        st.write(f"- {labels[1]}: **{prob[1]:.2f}**")
        st.write(f"- {labels[2]}: **{prob[2]:.2f}**")

        # Triage cues
        if prob[2] >= HEMORRHAGE_THRESHOLD:
            st.error("High hemorrhage probability — prioritize urgent review."
                     if lang == "English" else "احتمال مرتفع للنزف — أولوية قصوى للمراجعة.")
        elif prob[1] >= ISCHEMIC_THRESHOLD:
            st.warning("Possible ischemic stroke — consider urgent clinical correlation."
                       if lang == "English" else "سكتة إقفارية محتملة — يُنصح بمراجعة سريرية عاجلة.")

        # Grad-CAM (guarded) + compositor
        try:
            cam = gradcam(model, x, target_class=pred_idx)
            overlay = compose_heatmap(rgb, cam, alpha=0.4)
            st.image(overlay, caption="Attention heatmap (Grad-CAM)", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")

    except Exception as e:
        st.error(("Inference error: " + str(e)) if lang == "English" else ("خطأ في الاستدلال: " + str(e)))
    finally:
        if 'x' in locals(): del x
        if 'logits' in locals(): del logits
        if 'prob' in locals(): del prob
        import gc; gc.collect()

# ---------- Feature rows ----------
def feature_row(icon: Path, title_en: str, title_ar: str, desc_en: str, desc_ar: str) -> None:
    t = title_en if lang == "English" else title_ar
    d = desc_en if lang == "English" else desc_ar
    c1, c2 = st.columns([1, 12])
    with c1:
        if icon.exists():
            st.image(str(icon), width=40)
    with c2:
        st.markdown(f"**{t}** — {d}")

st.subheader("Key Features" if lang == "English" else "الميزات الأساسية")
feature_row(ICON_TIMER, "Faster triage", "فرز أسرع",
            "Meets ≤25-min door-to-CT KPI.", "يلبي مؤشر الأداء ≤ 25 دقيقة من الوصول إلى الأشعة.")
feature_row(ICON_BRAIN, "CT-first workflow", "مسار عمل يعتمد على الأشعة المقطعية أولًا",
            "Designed for non-contrast head CT in Oman EDs.", "مصمم للأشعة المقطعية دون تباين في أقسام الطوارئ بعُمان.")
feature_row(ICON_AI, "Decision support", "دعم القرار",
            "Grad-CAM shows where the model focuses.", "يوضح Grad-CAM مناطق اهتمام النموذج.")

# ---------- Footer ----------
st.markdown("<hr style='opacity:.3;'>", unsafe_allow_html=True)
footer = ("© 2025 Developed by Arwa Alghafri — Educational Prototype"
          if lang == "English" else "© ٢٠٢٥ تم تطويره بواسطة أروى الغافرية — نموذج تعليمي")
st.caption(footer)
