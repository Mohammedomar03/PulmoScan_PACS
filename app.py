import os
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import streamlit as st
from fpdf import FPDF
import tempfile
from datetime import datetime
from tensorflow.keras.applications.densenet import preprocess_input

# ==============================================================================
# 1. PAGE CONFIGURATION & HOSPITAL PACS STYLING
# ==============================================================================
st.set_page_config(
    page_title="PulmoScan PACS - Thoracic Radiography Suite",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Professional PACS Dark/Slate Theme with High-Contrast Text
st.markdown("""
<style>
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    
    .pacs-header {
        background: #0f172a;
        padding: 1.25rem 1.75rem;
        border-radius: 8px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #0284c7;
    }
    .pacs-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
    .pacs-header p { margin: 0.2rem 0 0 0; color: #94a3b8; font-size: 0.88rem; }

    .clinical-callout {
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border-left: 5px solid;
    }
    .callout-covid { background-color: #fef2f2; border-color: #dc2626; color: #991b1b; }
    .callout-opacity { background-color: #fff7ed; border-color: #ea580c; color: #9a3412; }
    .callout-viral { background-color: #fefce8; border-color: #ca8a04; color: #854d0e; }
    .callout-normal { background-color: #f0fdf4; border-color: #16a34a; color: #166534; }
    
    .rec-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 1.1rem;
        margin-top: 1rem;
        color: #0f172a !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .rec-box strong {
        color: #0369a1 !important;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["COVID-19", "Lung Opacity", "Normal", "Viral Pneumonia"]

# ==============================================================================
# 2. MODEL MANAGEMENT & LOCAL CLINICAL KNOWLEDGE BASE
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "PulmoScan PACS - Thoracic Imaging Suite.keras")

@st.cache_resource
def load_radiology_model():
    """Loads trained model from models subfolder."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_radiology_model():
    """Loads the trained Keras CNN model with automatic path fallback."""
    if not os.path.exists(MODEL_PATH):
        fallback = os.path.join(BASE_DIR, "best_densenet121_phase2.keras")
        if os.path.exists(fallback):
            return tf.keras.models.load_model(fallback)
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def generate_local_clinical_summary(primary_finding, confidence):
    """Generates a static medical response based strictly on the model result."""
    if "COVID" in primary_finding:
        features = "Peripheral and bilateral ground-glass opacities, lower zone predominance, multi-lobar involvement."
        steps = "Order RT-PCR confirmation test, isolate patient, monitor SpO2 levels, and evaluate inflammatory markers."
    elif "Opacity" in primary_finding:
        features = "Focal consolidation, air bronchograms, or localized opacification suggesting pulmonary parenchymal involvement."
        steps = "Correlate with clinical symptoms (fever, sputum), consider blood cultures, and evaluate for bacterial pneumonia."
    elif "Viral" in primary_finding:
        features = "Diffuse interstitial markings, peribronchial thickening, bilateral patchy infiltrates."
        steps = "Order viral respiratory panel, evaluate oxygen saturation, and provide supportive respiratory therapy."
    else:
        features = "Clear lung fields, normal cardiomegaly ratio, well-defined costophrenic angles, no focal consolidation."
        steps = "No acute cardiopulmonary findings. Continue standard monitoring if clinical symptoms persist."

    return f"""<strong>1. Primary Radiological Features Expected:</strong><br>{features}<br><br><strong>2. Recommended Clinical Evaluation Steps:</strong><br>{steps}"""

# ==============================================================================
# 3. PREPROCESSING & GRAD-CAM FEATURE LOCALIZATION
# ==============================================================================
def preprocess_radiograph(pil_img, target_size=(299, 299)):
    """Preprocesses input image into tensor matching DenseNet model requirements."""
    # 1. Standardize PIL image to 3-channel RGB and resize
    img = pil_img.convert("RGB").resize(target_size)

    # 2. Convert to NumPy float32 array (0 - 255 range)
    img_array = np.array(img, dtype=np.float32)

    # 3. Add batch dimension -> shape becomes (1, 299, 299, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # 4. Apply DenseNet preprocessing on the batch
    img_preprocessed = preprocess_input(img_batch)

    return img_preprocessed

def compute_feature_heatmap(model, img_tensor, class_idx):
    """Computes class activation maps for feature localization."""
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
                
        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception:
        return None

def generate_visual_overlay(original_img, heatmap, alpha=0.4, colormap_choice=cv2.COLORMAP_JET):
    """Overlays activation heatmaps on raw radiographs with selectable colormaps."""
    orig_np = np.array(original_img.resize((299, 299)).convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap_choice)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig_np, 1 - alpha, heatmap_colored, alpha, 0)

# ==============================================================================
# 4. PROFESSIONAL CONSULTATION REPORT GENERATOR
# ==============================================================================
def generate_radiology_report(patient_info, primary_finding, probability, all_probs, original_img):
    """Generates structured PACS radiology consultation PDF report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 8, "PULMOSCAN RADIOLOGY WORKSTATION", ln=1, align='L')
    pdf.set_font("helvetica", '', 9)
    pdf.cell(0, 5, "Department of Diagnostic Imaging | Thoracic Consultation Summary", ln=1, align='L')
    pdf.line(10, 24, 200, 24)
    
    # Patient Info
    pdf.ln(6)
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(40, 6, "Patient Identifier:")
    pdf.set_font("helvetica", '', 10)
    pdf.cell(50, 6, str(patient_info['id']))
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(30, 6, "Accession Date:")
    pdf.set_font("helvetica", '', 10)
    pdf.cell(50, 6, str(patient_info['date']), ln=1)
    
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(40, 6, "Ordering Physician:")
    pdf.set_font("helvetica", '', 10)
    pdf.cell(50, 6, str(patient_info['physician']), ln=1)
    
    pdf.line(10, 42, 200, 42)
    pdf.ln(6)
    
    # Findings
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 7, "IMPRESSION & FINDINGS", ln=1)
    pdf.set_font("helvetica", '', 11)
    pdf.cell(0, 6, f"Primary Classification: {primary_finding}", ln=1)
    pdf.cell(0, 6, f"Confidence Likelihood: {probability:.2f}%", ln=1)
    
    pdf.ln(4)
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(0, 6, "Differential Probability Distribution:", ln=1)
    pdf.set_font("helvetica", '', 9)
    for name, prob in zip(CLASS_NAMES, all_probs):
        pdf.cell(0, 5, f" - {name}: {prob*100:.2f}%", ln=1)

    # Image Snapshot
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "pacs_report_snap.jpg")
    original_img.resize((180, 180)).save(temp_path)
    
    pdf.ln(4)
    pdf.image(temp_path, x=65, w=80)
    
    pdf.ln(8)
    pdf.set_font("helvetica", 'I', 8)
    pdf.multi_cell(0, 4, "NOTICE: Computer-assisted screening report. AI outputs can be inaccurate or incorrect. Consultation with a licensed medical doctor is required.")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return bytes(pdf.output(dest='S').encode('latin1'))

# ==============================================================================
# 5. WORKSPACE & SIDEBAR INTERFACE
# ==============================================================================
st.markdown("""
<div class="pacs-header">
    <h1>🩻 PulmoScan PACS - Thoracic Imaging Suite</h1>
    <p>Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)

model = load_radiology_model()

if model is None:
    st.error("🚨 **System Offline:** Model weights file missing from `./model/` directory.")
    st.stop()

with st.sidebar:
    st.subheader("📌 Accession & Patient Details")
    patient_id = st.text_input("Accession Number", value="ACC-2026-8812")
    ordering_md = st.text_input("Ordering Clinician", value="Dr. M. Vance, MD")
    
    st.markdown("---")
    st.subheader("🎛️ Image Windowing")
    contrast_val = st.slider("Contrast Windowing", 0.5, 2.0, 1.0, 0.1)
    brightness_val = st.slider("Level / Brightness", 0.5, 2.0, 1.0, 0.1)
    invert_grayscale = st.checkbox("Invert Grayscale")
    
    st.markdown("---")
    st.subheader("🎨 Heatmap Palette")
    cmap_selection = st.selectbox("Palette", ["JET (Standard)", "BONE", "VIRIDIS", "PLASMA"])

cmap_map = {
    "JET (Standard)": cv2.COLORMAP_JET,
    "BONE": cv2.COLORMAP_BONE,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "PLASMA": cv2.COLORMAP_PLASMA
}

col_viewer, col_analysis = st.columns([1, 1.1], gap="large")

with col_viewer:
    st.markdown("### 📥 Image Acquisition")
    uploaded_file = st.file_uploader("Load Radiograph (PNG, JPG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file).convert("RGB")
        
        enhancer = ImageEnhance.Contrast(raw_img)
        adjusted_img = enhancer.enhance(contrast_val)
        enhancer = ImageEnhance.Brightness(adjusted_img)
        adjusted_img = enhancer.enhance(brightness_val)

        if invert_grayscale:
            adjusted_np = np.array(adjusted_img)
            adjusted_img = Image.fromarray(255 - adjusted_np)

        st.image(adjusted_img, caption="Processed Radiograph", use_container_width=True)
    else:
        st.info("👈 Upload a chest radiograph to start evaluation.")

with col_analysis:
    st.markdown("### 📊 Diagnostic Evaluation")
    
    if uploaded_file is not None:
        if st.button("🔬 Run Image Analysis", type="primary"):
            with st.spinner("Analyzing radiograph features..."):
                img_tensor = preprocess_radiograph(adjusted_img)
                predictions = model.predict(img_tensor)[0]
                
                top_idx = int(np.argmax(predictions))
                primary_finding = CLASS_NAMES[top_idx]
                confidence_score = float(predictions[top_idx]) * 100

                if "COVID" in primary_finding:
                    css_class = "callout-covid"
                    icon = "🦠"
                elif "Opacity" in primary_finding:
                    css_class = "callout-opacity"
                    icon = "⚠️"
                elif "Viral" in primary_finding:
                    css_class = "callout-viral"
                    icon = "🤒"
                else:
                    css_class = "callout-normal"
                    icon = "✅"

                st.markdown(f"""
                <div class="clinical-callout {css_class}">
                    {icon} <strong>PRIMARY IMPRESSION: {primary_finding.upper()}</strong><br>
                    Calculated Probability: {confidence_score:.2f}%
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📈 Differential Likelihoods")
                for name, prob in zip(CLASS_NAMES, predictions):
                    st.write(f"**{name}**: `{prob*100:.2f}%`")
                    st.progress(float(prob))

                heatmap = compute_feature_heatmap(model, img_tensor, top_idx)
                if heatmap is not None:
                    st.markdown("#### 🔍 Feature Localization Overlay")
                    selected_cmap = cmap_map[cmap_selection]
                    overlay_img = generate_visual_overlay(adjusted_img, heatmap, colormap_choice=selected_cmap)
                    
                    cam_c1, cam_c2 = st.columns(2)
                    with cam_c1:
                        st.image(heatmap, caption="Relative Intensity", use_container_width=True, clamp=True)
                    with cam_c2:
                        st.image(overlay_img, caption="Anatomical Overlay", use_container_width=True)

                # Cleaned Clinical Summary Box
                st.markdown("#### 📋 Clinical Summary & Guidance")
                
                clinical_summary = generate_local_clinical_summary(primary_finding, confidence_score)

                summary_html = f"""<div class="rec-box">
<strong>📋 Automated AI Response Summary:</strong><br><br>
{clinical_summary}
<hr style="margin: 12px 0; border: 0; border-top: 1px solid #cbd5e1;">
<span style="font-size: 0.82rem; color: #dc2626; font-weight: 600;">
⚠️ DISCLAIMER: AI outputs can be inaccurate or incorrect. This analysis is for decision support only. You must consult a licensed medical doctor or physician for an official diagnosis and treatment plan.
</span>
</div>"""

                st.markdown(summary_html, unsafe_allow_html=True)

                # PDF Export
                patient_data = {
                    'id': patient_id,
                    'physician': ordering_md,
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                pdf_report = generate_radiology_report(patient_data, primary_finding, confidence_score, predictions, adjusted_img)
                
                st.markdown("---")
                st.download_button(
                    label="📄 Export Consultation Report (PDF)",
                    data=pdf_report,
                    file_name=f"PACS_Report_{patient_id}.pdf",
                    mime="application/pdf"
                )