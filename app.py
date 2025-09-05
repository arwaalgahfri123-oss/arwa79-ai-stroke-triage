diff --git a/app.py b/app.py
index abe5732527c99c308c17c88e2b683c1f105f50f2..ed7e8afe6edc8effe5961419d07c17bae5a5de03 100644
--- a/app.py
+++ b/app.py
@@ -1,177 +1,322 @@
-diff --git a/app.py b/app.py
-index d940786..a13b7aa 100644
---- a/app.py
-+++ b/app.py
-@@ -14,6 +14,31 @@ from torchvision import models
- from utils_ct import load_image_any, apply_brain_window, gradcam  # helper funcs
- 
-+# ---------- 13" Mac-friendly CSS & RTL helper ----------
-+st.markdown("""
-+<style>
-+.block-container {padding-top: 1.2rem; padding-bottom: 3rem; padding-left: 2.0rem; padding-right: 2.0rem;}
-+html, body, [class^="css"] {font-size: 16px;}
-+[data-testid="stMetricLabel"] {font-size: 0.95rem; opacity: 0.9;}
-+.stButton button {width: 100%; padding: 0.6rem 0.8rem;}
-+.st-emotion-cache-1kyxreq, .stCaption {font-size: 0.9rem; opacity: 0.85;}
-+[data-baseweb="tab-list"] {gap: 0.5rem;}
-+.rtl {direction: rtl; text-align: right;}
-+</style>
-+""", unsafe_allow_html=True)
-+
-+def rtl_wrap(html: str, is_arabic: bool) -> str:
-+    """Wrap HTML in RTL container if Arabic is active."""
-+    return f"<div class='rtl'>{html}</div>" if is_arabic else html
-+
-@@ -63,12 +88,12 @@ col1, col2 = st.columns([1, 5])
- with col1:
-     if LOGO.exists():
-         st.image(str(LOGO), width=80)
- with col2:
--    st.markdown(
--        "<h2 style='margin-bottom:0; color:#3EC8E0;'>AI Stroke Triage — Oman</h2>"
--        "<p style='opacity:.85; color:#EAEAEA;'>Bilingual demo • ≤25-min target • Vision 2040</p>"
--        f"<p style='opacity:.7; font-size:0.9rem; color:#EAEAEA;'>{dev_text}</p>",
--        unsafe_allow_html=True,
--    )
-+    title_html = (
-+        "<h2 style='margin-bottom:0; color:#3EC8E0;'>AI Stroke Triage — Oman</h2>"
-+        "<p style='opacity:.85; color:#EAEAEA;'>Bilingual demo • ≤25-min target • Vision 2040</p>"
-+        f"<p style='opacity:.7; font-size:0.9rem; color:#EAEAEA;'>{dev_text}</p>"
-+    )
-+    st.markdown(rtl_wrap(title_html, lang == "العربية"), unsafe_allow_html=True)
- st.caption(TXT["note"][lang])
- st.divider()
- 
-@@ -142,7 +167,7 @@ if uploaded is not None:
-             img_win = None
- 
- # ---------- Layout ----------
--left, right = st.columns([2, 1])
-+left, right = st.columns([1.6, 1.0], vertical_alignment="top")
- 
- with left:
-     if img_win is not None:
-@@ -160,6 +185,10 @@ with right:
-         stop_clicked = st.button(TXT["stop"][lang], use_container_width=True)
-     if start_clicked:
-         start_timer()
-     if stop_clicked:
-         stop_timer()
-+    # DEMO badge if no custom checkpoint
-+    if not trained:
-+        st.warning("DEMO mode: running on ImageNet weights (no custom checkpoint found)."
-+                   if lang=="English" else "وضع العرض: يعمل بأوزان ImageNet (لا يوجد نموذج مخصص).")
- 
- # ---------- Inference ----------
- if img_win is not None and start_clicked:
-@@ -187,25 +216,44 @@ if img_win is not None and start_clicked:
-         overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
- 
--        st.subheader(TXT["results"][lang])
--        st.write(f"**{pred_label}**")
--        st.write(f"- {labels[0]}: **{prob[0]:.2f}**")
--        st.write(f"- {labels[1]}: **{prob[1]:.2f}**")
--        st.write(f"- {labels[2]}: **{prob[2]:.2f}**")
--
--        # Triage cues use configurable thresholds
--        if prob[2] >= HEMORRHAGE_THRESHOLD:
--            st.error("High hemorrhage probability — prioritize urgent review."
--                     if lang == "English" else "احتمال مرتفع للنزف — أولوية قصوى للمراجعة.")
--        elif prob[1] >= ISCHEMIC_THRESHOLD:
--            st.warning("Possible ischemic stroke — consider urgent clinical correlation."
--                       if lang == "English" else "سكتة إقفارية محتملة — يُنصح بمراجعة سريرية عاجلة.")
--
--        st.image(overlay, caption="Attention heatmap (Grad-CAM)", use_container_width=True)
-+        # ----- Results in tabs (Overview / Heatmap / Details) -----
-+        tab_overview, tab_heatmap, tab_details = st.tabs([
-+            "Overview" if lang=="English" else "نظرة عامة",
-+            "Heatmap" if lang=="English" else "خريطة الاهتمام",
-+            "Details" if lang=="English" else "تفاصيل",
-+        ])
-+
-+        with tab_overview:
-+            st.subheader(TXT["results"][lang])
-+            st.write(f"**{pred_label}**")
-+            st.write(f"- {labels[0]}: **{prob[0]:.2f}**")
-+            st.write(f"- {labels[1]}: **{prob[1]:.2f}**")
-+            st.write(f"- {labels[2]}: **{prob[2]:.2f}**")
-+            # Triage cues use configurable thresholds
-+            if prob[2] >= HEMORRHAGE_THRESHOLD:
-+                st.error("High hemorrhage probability — prioritize urgent review."
-+                         if lang == "English" else "احتمال مرتفع للنزف — أولوية قصوى للمراجعة.")
-+            elif prob[1] >= ISCHEMIC_THRESHOLD:
-+                st.warning("Possible ischemic stroke — consider urgent clinical correlation."
-+                           if lang == "English" else "سكتة إقفارية محتملة — يُنصح بمراجعة سريرية عاجلة.")
-+
-+        with tab_heatmap:
-+            st.image(
-+                overlay,
-+                caption="Attention heatmap (Grad-CAM)" if lang=="English" else "خريطة الاهتمام (Grad-CAM)",
-+                use_container_width=True,
-+            )
-+
-+        with tab_details:
-+            st.caption(f"Device: {device} • Weights: {'Custom' if trained else 'DEMO'}")
-+            st.caption(f"Thresholds — Ischemic: {ISCHEMIC_THRESHOLD:.2f} • Hemorrhage: {HEMORRHAGE_THRESHOLD:.2f}")
- 
-     except Exception as e:
-         st.error(("Inference error: " + str(e)) if lang == "English" else ("خطأ في الاستدلال: " + str(e)))
-     finally:
-         # minimal cleanup
-         if 'x' in locals(): del x
-         if 'logits' in locals(): del logits
-         if 'prob' in locals(): del prob
-         import gc; gc.collect()
- 
- # ---------- Optional: simple feature rows using your icons ----------
-@@ -218,18 +266,28 @@ def feature_row(
-     with c2:
-         st.markdown(f"**{t}** — {d}")
- 
--st.subheader("Key Features" if lang == "English" else "الميزات الأساسية")
--feature_row(
--    ICON_TIMER,
--    "Faster triage", "فرز أسرع",
--    "Meets ≤25-min door-to-CT KPI.", "يلبي مؤشر الأداء ≤ 25 دقيقة من الوصول إلى الأشعة."
--)
--feature_row(
--    ICON_BRAIN,
--    "CT-first workflow", "مسار عمل يعتمد على الأشعة المقطعية أولًا",
--    "Designed for non-contrast head CT in Oman EDs.", "مصمم للأشعة المقطعية دون تباين في أقسام الطوارئ بعُمان."
--)
--feature_row(
--    ICON_AI,
--    "Decision support", "دعم القرار",
--    "Grad-CAM shows where the model focuses.", "يوضح Grad-CAM مناطق اهتمام النموذج."
--)
-+st.subheader("Key Features" if lang == "English" else "الميزات الأساسية")
-+fc1, fc2 = st.columns(2)
-+with fc1:
-+    feature_row(
-+        ICON_TIMER,
-+        "Faster triage", "فرز أسرع",
-+        "Meets ≤25-min door-to-CT KPI.", "يلبي مؤشر الأداء ≤ 25 دقيقة من الوصول إلى الأشعة."
-+    )
-+with fc2:
-+    feature_row(
-+        ICON_BRAIN,
-+        "CT-first workflow", "مسار عمل يعتمد على الأشعة المقطعية أولًا",
-+        "Designed for non-contrast head CT in Oman EDs.", "مصمم للأشعة المقطعية دون تباين في أقسام الطوارئ بعُمان."
-+    )
-+with fc1:
-+    feature_row(
-+        ICON_AI,
-+        "Decision support", "دعم القرار",
-+        "Grad-CAM shows where the model focuses.", "يوضح Grad-CAM مناطق اهتمام النموذج."
-+    )
-diff --git a/.streamlit/config.toml b/.streamlit/config.toml
-new file mode 100644
---- /dev/null
-+++ b/.streamlit/config.toml
-@@ -0,0 +1,7 @@
-+[theme]
-+primaryColor = "#3EC8E0"
-+backgroundColor = "#0E1117"
-+secondaryBackgroundColor = "#161A23"
-+textColor = "#EAEAEA"
-+font = "sans serif"
+# app.py
+# Streamlit AI Stroke Triage — Oman Prototype
+# Educational demo only. Not for clinical use.
+
+import os
+import time
+from pathlib import Path
+
+import cv2
+import numpy as np
+import streamlit as st
+import torch
+import torch.nn as nn
+import torchvision.transforms as T
+from PIL import Image
+from torchvision import models
+
+from utils_ct import (
+    load_image_any, apply_brain_window, apply_window_preset,
+    blur_score_variance_of_laplacian, extract_dicom_meta, gradcam
+)
+
+# ---------- 13" Mac-friendly CSS & RTL helper ----------
+st.markdown("""
+<style>
+.block-container {padding-top: 1.2rem; padding-bottom: 3rem; padding-left: 2.0rem; padding-right: 2.0rem; max-width: 1000px; margin: 0 auto;}
+html, body, [class^="css"] {font-size: 15px;}
+[data-testid="stMetricLabel"] {font-size: 0.95rem; opacity: 0.9;}
+.stButton button {width: 100%; padding: 0.6rem 0.8rem;}
+.st-emotion-cache-1kyxreq, .stCaption {font-size: 0.9rem; opacity: 0.85;}
+[data-baseweb="tab-list"] {gap: 0.5rem;}
+.rtl {direction: rtl; text-align: right;}
+</style>
+""", unsafe_allow_html=True)
+
+def rtl_wrap(html: str, is_arabic: bool) -> str:
+    """Wrap HTML in RTL container if Arabic is active."""
+    return f"<div class='rtl'>{html}</div>" if is_arabic else html
+
+# ---------- Assets ----------
+LOGO = Path("logo.png")
+ICON_TIMER = Path("icon_timer.png")
+ICON_BRAIN = Path("brain_scan.png")
+ICON_AI = Path("ai.png")
+
+# ---------- Config ----------
+MODEL_PATH = os.getenv("MODEL_PATH", "models/stroke_3class.pt")  # 3-class checkpoint when you train
+PAGE_TITLE = "AI Stroke Triage — Oman Prototype"
+st.set_page_config(page_title=PAGE_TITLE, page_icon="logo.png", layout="wide")
+
+# ---------- Language toggle ----------
+lang = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
+
+# Prob thresholds (sidebar sliders so they’re tunable & persistent)
+st.sidebar.markdown("---")
+st.sidebar.markdown("**Triage thresholds**" if lang == "English" else "**عتبات الفرز**")
+HEMORRHAGE_THRESHOLD = st.sidebar.slider(
+    "Hemorrhage alert" if lang == "English" else "تنبيه النزف",
+    min_value=0.30, max_value=0.95, value=st.session_state.get("HEM_THR", 0.60), step=0.01
+)
+ISCHEMIC_THRESHOLD = st.sidebar.slider(
+    "Ischemic alert" if lang == "English" else "تنبيه الإقفار",
+    min_value=0.30, max_value=0.95, value=st.session_state.get("ISC_THR", 0.60), step=0.01
+)
+st.session_state["HEM_THR"] = HEMORRHAGE_THRESHOLD
+st.session_state["ISC_THR"] = ISCHEMIC_THRESHOLD
+
+# 3-class labels (bilingual)
+CLASS_NAMES = {
+    "English": ["Normal / No acute stroke", "Ischemic stroke (suspected)", "Hemorrhagic stroke"],
+    "العربية": ["طبيعي / لا سكتة حادة", "سكتة إقفارية (مشتبه بها)", "سكتة نزفية"],
+}
+
+TXT = {
+    "title": {"English": "AI Stroke Triage — Oman Prototype",
+              "العربية": "نموذج أولي لفرز السكتة الدماغية بالذكاء الاصطناعي — عُمان"},
+    "upload": {"English": "Upload axial head CT (DICOM/PNG/JPG)",
+               "العربية": "حمّل صورة الأشعة المقطعية للرأس (DICOM/PNG/JPG)"},
+    "run": {"English": "Run Triage", "العربية": "تشغيل الفرز"},
+    "stop": {"English": "Stop", "العربية": "إيقاف"},
+    "note": {"English": "Educational demo. Not for clinical use.",
+             "العربية": "عرض تعليمي فقط. ليس للاستخدام السريري."},
+    "kpi": {"English": "Triage timer (target ≤ 25 min)",
+            "العربية": "مؤقت الفرز (الهدف ≤ 25 دقيقة)"},
+    "results": {"English": "AI Findings", "العربية": "نتائج الذكاء الاصطناعي"},
+}
+
+# Bilingual developer credit
+dev_text = ("Developed by <b>Arwa Alghafri</b>"
+            if lang == "English"
+            else "تم تطويره بواسطة <b>أروى الغافرية</b>")
+
+# ---------- Header ----------
+col1, col2 = st.columns([1, 5])
+with col1:
+    if LOGO.exists():
+        st.image(str(LOGO), width=150)
+with col2:
+    title_html = (
+        "<h2 style='margin-bottom:0; color:#3EC8E0;'>AI Stroke Triage — Oman</h2>"
+        "<p style='opacity:.85; color:#EAEAEA;'>Bilingual demo • ≤25-min target • Vision 2040</p>"
+        f"<p style='opacity:.7; font-size:0.9rem; color:#EAEAEA;'>{dev_text}</p>"
+    )
+    st.markdown(rtl_wrap(title_html, lang == "العربية"), unsafe_allow_html=True)
+st.caption(TXT["note"][lang])
+st.divider()
+
+# ---------- Timer state & helpers ----------
+if "timer_running" not in st.session_state:
+    st.session_state.timer_running = False
+if "start_time" not in st.session_state:
+    st.session_state.start_time = None
+
+def start_timer() -> None:
+    st.session_state.start_time = time.time()
+    st.session_state.timer_running = True
+
+def stop_timer() -> None:
+    st.session_state.timer_running = False
+
+def read_timer() -> str:
+    if st.session_state.timer_running and st.session_state.start_time:
+        elapsed = time.time() - st.session_state.start_time
+        mm = int(elapsed // 60); ss = int(elapsed % 60)
+        return f"{mm:02d}:{ss:02d}"
+    return "00:00"
+
+# ---------- Model ----------
+@st.cache_resource
+def load_model(model_path: str) -> tuple[nn.Module, bool, torch.device]:
+    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
+    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
+    in_f = model.fc.in_features
+    model.fc = nn.Linear(in_f, 3)
+    trained = False
+    if os.path.exists(model_path):
+        try:
+            state = torch.load(model_path, map_location="cpu")
+            try:
+                model.load_state_dict(state, strict=True)
+                trained = True
+            except RuntimeError:
+                md = model.state_dict()
+                filtered = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
+                md.update(filtered)
+                model.load_state_dict(md, strict=False)
+                st.warning("Checkpoint partially loaded (shape mismatch). Running in partial/DEMO mode.")
+                trained = len(filtered) > 0
+        except Exception as e:
+            st.warning(f"Failed to load weights at {model_path}: {e}. Using ImageNet weights (DEMO).")
+    model.to(device).eval()
+    return model, trained, device
+
+model, trained, device = load_model(MODEL_PATH)
+st.sidebar.caption(f"Compute device: {'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}")
+
+transform = T.Compose([
+    T.Resize((256, 256)),
+    T.ToTensor(),
+    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
+])
+
+# ---------- File validation & pre-load image ----------
+allowed_ext = {"dcm", "png", "jpg", "jpeg"}
+uploaded = st.file_uploader(TXT["upload"][lang], type=list(allowed_ext))
+
+img_win = None
+is_dicom = False
+meta = {}
+if uploaded is not None:
+    ext = uploaded.name.split(".")[-1].lower()
+    is_dicom = (ext == "dcm")
+    try:
+        img_gray = load_image_any(uploaded)             # HxW uint8
+        img_win = apply_brain_window(img_gray)          # default brain window
+        # Optional: show subdural preset example (commented)
+        # img_win = apply_window_preset(img_gray, "subdural")
+        if is_dicom:
+            meta = extract_dicom_meta(uploaded)
+    except Exception:
+        st.error("Could not read this file. Please upload a valid head CT DICOM/PNG/JPG."
+                 if lang == "English" else "تعذر قراءة الملف. يرجى تحميل أشعة مقطعية للرأس بصيغة صحيحة.")
+        img_win = None
+
+# ---------- Layout ----------
+start_clicked = False
+left, right = st.columns([1.6, 1.0], vertical_alignment="top")
+
+with left:
+    if img_win is not None:
+        st.image(img_win, caption="Brain window", use_container_width=True, clamp=True)
+
+        # Blur/quality warning
+        score = blur_score_variance_of_laplacian(img_win)
+        if score < 80:
+            st.warning("Image appears blurred (low sharpness)."
+                       if lang == "English" else "تبدو الصورة غير واضحة (حدة منخفضة).")
+
+        # Optional metadata for DICOMs
+        if is_dicom and meta:
+            with st.expander("DICOM technical info" if lang == "English" else "معلومات تقنية عن DICOM"):
+                st.json(meta)
+    else:
+        st.info("Upload a CT image to begin." if lang == "English" else "قم بتحميل صورة الأشعة المقطعية للبدء.")
+
+with right:
+    if not trained:
+        st.info("Running in DEMO mode (no 3-class checkpoint found). "
+                "Add models/stroke_3class.pt for realistic performance.")
+    st.metric(TXT["kpi"][lang], value=read_timer(), delta="Target ≤ 25 min")
+    c1, c2 = st.columns(2)
+    with c1:
+        start_clicked = st.button(TXT["run"][lang], use_container_width=True)
+    with c2:
+        stop_clicked = st.button(TXT["stop"][lang], use_container_width=True)
+    if start_clicked:
+        start_timer()
+    if stop_clicked:
+        stop_timer()
+    # DEMO badge if no custom checkpoint
+    if not trained:
+        st.warning("DEMO mode: running on ImageNet weights (no custom checkpoint found)."
+                   if lang=="English" else "وضع العرض: يعمل بأوزان ImageNet (لا يوجد نموذج مخصص).")
+
+# ---------- Inference ----------
+if img_win is not None and start_clicked:
+    try:
+        rgb = cv2.cvtColor(img_win, cv2.COLOR_GRAY2RGB)
+        pil = Image.fromarray(rgb)
+        x = transform(pil).unsqueeze(0).to(device)
+
+        with torch.no_grad():
+            logits = model(x)
+            if logits.ndim != 2 or logits.shape[1] != 3:
+                raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
+            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (3,)
+
+        pred_idx = int(prob.argmax())
+        labels = CLASS_NAMES[lang]
+        pred_label = labels[pred_idx]
+
+        cam = gradcam(model, x, target_class=pred_idx)
+        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
+        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
+        overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
+
+        # ----- Results in tabs (Overview / Heatmap / Details) -----
+        tab_overview, tab_heatmap, tab_details = st.tabs([
+            "Overview" if lang=="English" else "نظرة عامة",
+            "Heatmap" if lang=="English" else "خريطة الاهتمام",
+            "Details" if lang=="English" else "تفاصيل",
+        ])
+
+        with tab_overview:
+            st.subheader(TXT["results"][lang])
+            st.write(f"**{pred_label}**")
+            st.write(f"- {labels[0]}: **{prob[0]:.2f}**")
+            st.write(f"- {labels[1]}: **{prob[1]:.2f}**")
+            st.write(f"- {labels[2]}: **{prob[2]:.2f}**")
+            if prob[2] >= HEMORRHAGE_THRESHOLD:
+                st.error("High hemorrhage probability — prioritize urgent review."
+                         if lang == "English" else "احتمال مرتفع للنزف — أولوية قصوى للمراجعة.")
+            elif prob[1] >= ISCHEMIC_THRESHOLD:
+                st.warning("Possible ischemic stroke — consider urgent clinical correlation."
+                           if lang == "English" else "سكتة إقفارية محتملة — يُنصح بمراجعة سريرية عاجلة.")
+
+        with tab_heatmap:
+            st.image(
+                overlay,
+                caption="Attention heatmap (Grad-CAM)" if lang=="English" else "خريطة الاهتمام (Grad-CAM)",
+                use_container_width=True,
+            )
+
+        with tab_details:
+            st.caption(f"Device: {device} • Weights: {'Custom' if trained else 'DEMO'}")
+            st.caption(f"Thresholds — Ischemic: {ISCHEMIC_THRESHOLD:.2f} • Hemorrhage: {HEMORRHAGE_THRESHOLD:.2f}")
+
+    except Exception as e:
+        st.error(("Inference error: " + str(e)) if lang == "English" else ("خطأ في الاستدلال: " + str(e)))
+    finally:
+        if 'x' in locals(): del x
+        if 'logits' in locals(): del logits
+        if 'prob' in locals(): del prob
+        import gc; gc.collect()
+
+# ---------- Feature rows ----------
+def feature_row(icon: Path, title_en: str, title_ar: str, desc_en: str, desc_ar: str) -> None:
+    t = title_en if lang == "English" else title_ar
+    d = desc_en if lang == "English" else desc_ar
+    c1, c2 = st.columns([1, 12])
+    with c1:
+        if icon.exists():
+            st.image(str(icon), width=40)
+    with c2:
+        st.markdown(f"**{t}** — {d}")
+
+st.subheader("Key Features" if lang == "English" else "الميزات الأساسية")
+fc1, fc2, fc3 = st.columns(3)
+with fc1:
+    feature_row(
+        ICON_TIMER,
+        "Faster triage", "فرز أسرع",
+        "Meets ≤25-min door-to-CT KPI.", "يلبي مؤشر الأداء ≤ 25 دقيقة من الوصول إلى الأشعة."
+    )
+with fc2:
+    feature_row(
+        ICON_BRAIN,
+        "CT-first workflow", "مسار عمل يعتمد على الأشعة المقطعية أولًا",
+        "Designed for non-contrast head CT in Oman EDs.", "مصمم للأشعة المقطعية دون تباين في أقسام الطوارئ بعُمان."
+    )
+with fc3:
+    feature_row(
+        ICON_AI,
+        "Decision support", "دعم القرار",
+        "Grad-CAM shows where the model focuses.", "يوضح Grad-CAM مناطق اهتمام النموذج."
+    )
+
+# ---------- Footer ----------
+st.markdown("<hr style='opacity:.3;'>", unsafe_allow_html=True)
+footer = ("© 2025 Developed by Arwa Alghafri — Educational Prototype"
+          if lang == "English" else "© ٢٠٢٥ تم تطويره بواسطة أروى الغافرية — نموذج تعليمي")
+st.caption(footer)

+font = "sans serif"
