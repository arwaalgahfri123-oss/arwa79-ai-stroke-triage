diff --git a/app.py b/app.py
index d940786..a13b7aa 100644
--- a/app.py
+++ b/app.py
@@ -14,6 +14,31 @@ from torchvision import models
 from utils_ct import load_image_any, apply_brain_window, gradcam  # helper funcs
 
+# ---------- 13" Mac-friendly CSS & RTL helper ----------
+st.markdown("""
+<style>
+.block-container {padding-top: 1.2rem; padding-bottom: 3rem; padding-left: 2.0rem; padding-right: 2.0rem;}
+html, body, [class^="css"] {font-size: 16px;}
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
@@ -63,12 +88,12 @@ col1, col2 = st.columns([1, 5])
 with col1:
     if LOGO.exists():
         st.image(str(LOGO), width=80)
 with col2:
-    st.markdown(
-        "<h2 style='margin-bottom:0; color:#3EC8E0;'>AI Stroke Triage — Oman</h2>"
-        "<p style='opacity:.85; color:#EAEAEA;'>Bilingual demo • ≤25-min target • Vision 2040</p>"
-        f"<p style='opacity:.7; font-size:0.9rem; color:#EAEAEA;'>{dev_text}</p>",
-        unsafe_allow_html=True,
-    )
+    title_html = (
+        "<h2 style='margin-bottom:0; color:#3EC8E0;'>AI Stroke Triage — Oman</h2>"
+        "<p style='opacity:.85; color:#EAEAEA;'>Bilingual demo • ≤25-min target • Vision 2040</p>"
+        f"<p style='opacity:.7; font-size:0.9rem; color:#EAEAEA;'>{dev_text}</p>"
+    )
+    st.markdown(rtl_wrap(title_html, lang == "العربية"), unsafe_allow_html=True)
 st.caption(TXT["note"][lang])
 st.divider()
 
@@ -142,7 +167,7 @@ if uploaded is not None:
             img_win = None
 
 # ---------- Layout ----------
-left, right = st.columns([2, 1])
+left, right = st.columns([1.6, 1.0], vertical_alignment="top")
 
 with left:
     if img_win is not None:
@@ -160,6 +185,10 @@ with right:
         stop_clicked = st.button(TXT["stop"][lang], use_container_width=True)
     if start_clicked:
         start_timer()
     if stop_clicked:
         stop_timer()
+    # DEMO badge if no custom checkpoint
+    if not trained:
+        st.warning("DEMO mode: running on ImageNet weights (no custom checkpoint found)."
+                   if lang=="English" else "وضع العرض: يعمل بأوزان ImageNet (لا يوجد نموذج مخصص).")
 
 # ---------- Inference ----------
 if img_win is not None and start_clicked:
@@ -187,25 +216,44 @@ if img_win is not None and start_clicked:
         overlay = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
 
-        st.subheader(TXT["results"][lang])
-        st.write(f"**{pred_label}**")
-        st.write(f"- {labels[0]}: **{prob[0]:.2f}**")
-        st.write(f"- {labels[1]}: **{prob[1]:.2f}**")
-        st.write(f"- {labels[2]}: **{prob[2]:.2f}**")
-
-        # Triage cues use configurable thresholds
-        if prob[2] >= HEMORRHAGE_THRESHOLD:
-            st.error("High hemorrhage probability — prioritize urgent review."
-                     if lang == "English" else "احتمال مرتفع للنزف — أولوية قصوى للمراجعة.")
-        elif prob[1] >= ISCHEMIC_THRESHOLD:
-            st.warning("Possible ischemic stroke — consider urgent clinical correlation."
-                       if lang == "English" else "سكتة إقفارية محتملة — يُنصح بمراجعة سريرية عاجلة.")
-
-        st.image(overlay, caption="Attention heatmap (Grad-CAM)", use_container_width=True)
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
+            # Triage cues use configurable thresholds
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
 
     except Exception as e:
         st.error(("Inference error: " + str(e)) if lang == "English" else ("خطأ في الاستدلال: " + str(e)))
     finally:
         # minimal cleanup
         if 'x' in locals(): del x
         if 'logits' in locals(): del logits
         if 'prob' in locals(): del prob
         import gc; gc.collect()
 
 # ---------- Optional: simple feature rows using your icons ----------
@@ -218,18 +266,28 @@ def feature_row(
     with c2:
         st.markdown(f"**{t}** — {d}")
 
-st.subheader("Key Features" if lang == "English" else "الميزات الأساسية")
-feature_row(
-    ICON_TIMER,
-    "Faster triage", "فرز أسرع",
-    "Meets ≤25-min door-to-CT KPI.", "يلبي مؤشر الأداء ≤ 25 دقيقة من الوصول إلى الأشعة."
-)
-feature_row(
-    ICON_BRAIN,
-    "CT-first workflow", "مسار عمل يعتمد على الأشعة المقطعية أولًا",
-    "Designed for non-contrast head CT in Oman EDs.", "مصمم للأشعة المقطعية دون تباين في أقسام الطوارئ بعُمان."
-)
-feature_row(
-    ICON_AI,
-    "Decision support", "دعم القرار",
-    "Grad-CAM shows where the model focuses.", "يوضح Grad-CAM مناطق اهتمام النموذج."
-)
+st.subheader("Key Features" if lang == "English" else "الميزات الأساسية")
+fc1, fc2 = st.columns(2)
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
+with fc1:
+    feature_row(
+        ICON_AI,
+        "Decision support", "دعم القرار",
+        "Grad-CAM shows where the model focuses.", "يوضح Grad-CAM مناطق اهتمام النموذج."
+    )
diff --git a/.streamlit/config.toml b/.streamlit/config.toml
new file mode 100644
--- /dev/null
+++ b/.streamlit/config.toml
@@ -0,0 +1,7 @@
+[theme]
+primaryColor = "#3EC8E0"
+backgroundColor = "#0E1117"
+secondaryBackgroundColor = "#161A23"
+textColor = "#EAEAEA"
+font = "sans serif"
