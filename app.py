"""
Maize Disease Detection System — Streamlit Web Application
============================================================
TechCrush AI/ML Bootcamp · Cohort 5 · Capstone Project

Architecture : EfficientNetB0 (Transfer Learning)
Framework    : Streamlit
Input        : Maize leaf photograph (JPG / JPEG / PNG)
Output       : Disease class, confidence score, treatment and prevention advice

Classes detected:
    0 — Blight         (Exserohilum turcicum / Bipolaris maydis)
    1 — Common Rust    (Puccinia sorghi)
    2 — Gray Leaf Spot (Cercospora zeae-maydis)
    3 — Healthy

Note: Class indices follow the alphabetical ordering produced by Keras
      ImageDataGenerator.flow_from_directory().

AI Assistance: Portions of this code were developed with the assistance of
               Claude (Anthropic). All AI-generated sections have been reviewed,
               understood, and adapted specifically for this project.
               — TechCrush Cohort 5 Capstone Guidelines §4.1
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io
import os

# ── Third-party libraries ─────────────────────────────────────────────────────
import numpy as np
from PIL import Image
import tensorflow as tf                                    # noqa: F401 (required by Keras backend)
import keras
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras import layers, models

# ── Streamlit (UI framework) ──────────────────────────────────────────────────
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title="Maize Disease Detection System",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global constants ──────────────────────────────────────────────────────────

# Class labels must match the alphabetical ordering produced by Keras's
# ImageDataGenerator.flow_from_directory() during training.
#   Index 0 → Blight | 1 → Common_rust | 2 → Gray_leaf_spot | 3 → Healthy
CLASS_NAMES   = ["Blight", "Common_rust", "Gray_leaf_spot", "Healthy"]

# Human-readable versions of CLASS_NAMES used for display only.
DISPLAY_NAMES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

# Model input resolution (pixels × pixels). Must match training configuration.
IMG_SIZE      = (224, 224)

# Path to the saved EfficientNetB0 weight file (relative to app.py).
WEIGHTS_FILE  = "model_weights.weights.h5"

# Minimum confidence (%) below which a low-confidence warning is shown.
CONF_WARN     = 60

# ── Disease knowledge base ────────────────────────────────────────────────────
# Each key is the display name of the disease class.
# Keys per entry: emoji, color, glow (CSS shadow), severity, severity_color,
#                 description, cause, treatment, prevention.
DISEASE_INFO = {
    "Blight": {
        "emoji": "🟤", "color": "#ff3d00", "glow": "0 0 18px #ff3d0088",
        "severity": "High", "severity_color": "#ff3d00",
        "description": "Large irregular lesions with water-soaked margins that turn brown. Caused by Exserohilum turcicum or Bipolaris maydis. Can destroy entire canopies if left untreated.",
        "cause": "Fungal — Exserohilum turcicum / Bipolaris maydis.",
        "treatment": "• Apply azoxystrobin or tebuconazole fungicide.\n• Remove and burn all infected plant material.\n• Improve field drainage.\n• Rotate to a non-host crop next season.",
        "prevention": "• Plant blight-resistant hybrid varieties.\n• Avoid excess nitrogen fertiliser.\n• Practice clean tillage after harvest.\n• Scout fields regularly during humid seasons.",
    },
    "Common Rust": {
        "emoji": "🟠", "color": "#ff6d00", "glow": "0 0 18px #ff6d0088",
        "severity": "Moderate", "severity_color": "#ffab40",
        "description": "Orange-brown powdery pustules appear on both leaf surfaces. Caused by Puccinia sorghi. Spreads rapidly through wind-borne spores in cool, humid conditions.",
        "cause": "Fungal — Puccinia sorghi. Wind-borne spores.",
        "treatment": "• Apply mancozeb (2g/L) or propiconazole (1mL/L).\n• Repeat spray every 10–14 days.\n• Remove heavily infected leaves early.\n• Avoid overhead irrigation.",
        "prevention": "• Plant rust-resistant hybrids.\n• Avoid late-season planting.\n• Monitor crops during warm humid periods.",
    },
    "Gray Leaf Spot": {
        "emoji": "⚠️", "color": "#ff9100", "glow": "0 0 18px #ff910088",
        "severity": "High", "severity_color": "#ff9100",
        "description": "Long, narrow rectangular lesions running parallel to leaf veins. Caused by Cercospora zeae-maydis. Considered the most yield-limiting foliar disease of maize globally.",
        "cause": "Fungal — Cercospora zeae-maydis. Survives on crop residue.",
        "treatment": "• Apply azoxystrobin or pyraclostrobin.\n• Repeat treatment every 14 days.\n• Reduce canopy density through spacing.\n• Avoid overhead irrigation at night.",
        "prevention": "• Plant resistant hybrid varieties.\n• Till residue into soil between seasons.\n• Scout fields after morning dew periods.",
    },
    "Healthy": {
        "emoji": "✅", "color": "#00e676", "glow": "0 0 18px #00e67688",
        "severity": "None", "severity_color": "#00e676",
        "description": "No disease detected. Leaves show uniform green colour with no lesion, pustule or spot features. Plant is growing normally.",
        "cause": "N/A — plant is healthy.",
        "treatment": "• Maintain proper plant spacing for airflow.\n• Apply balanced NPK fertiliser as needed.\n• Continue scouting every 7–10 days.\n• Avoid waterlogging the root zone.",
        "prevention": "• Use certified disease-resistant seed varieties.\n• Rotate crops each season to break disease cycles.\n• Remove and dispose of crop debris after harvest.",
    },
}

# Maps raw CLASS_NAMES labels (underscore format from training) to the
# DISEASE_INFO dictionary keys (display format with spaces).
CLASS_TO_INFO = {
    "Blight":         "Blight",
    "Common_rust":    "Common Rust",
    "Gray_leaf_spot": "Gray Leaf Spot",
    "Healthy":        "Healthy",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
# Injected via st.markdown(unsafe_allow_html=True) to apply the project's
# black-and-orange neon theme over Streamlit's default styles.
# Assisted by Claude (Anthropic); adapted and customised for this project.
CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #0a0a0a !important; color: #f0f0f0 !important; }
  .stApp { background-color: #0a0a0a !important; }
  .block-container { padding-top: 1.5rem !important; max-width: 1200px; }
  #MainMenu, footer { visibility: hidden; }
  section[data-testid="stSidebar"] { background: #111111 !important; border-right: 2px solid #ff6d00; }
  section[data-testid="stSidebar"] * { color: #cccccc !important; }
  .sb-card { background: #151515; border-left: 3px solid #ff6d00; border-radius: 6px; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.82rem; }
  .sb-title { font-size: 1.1rem; font-weight: 800; background: linear-gradient(90deg, #ff6d00, #ffab40); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero { background: linear-gradient(135deg, #0f0f0f 0%, #1a0a00 60%, #0f0f0f 100%); border: 1px solid #2a1500; border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; }
  .hero-tag { display: inline-block; background: #ff6d0022; border: 1px solid #ff6d0055; color: #ff6d00 !important; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.8rem; }
  .hero h1 { font-size: 2.2rem; font-weight: 900; margin: 0 0 0.4rem; line-height: 1.2; color: #ffffff !important; }
  .hero h1 span { background: linear-gradient(90deg, #ff6d00, #ffab40); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero p { font-size: 0.95rem; color: #888 !important; margin: 0; }
  .stat-box { background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 0.9rem; text-align: center; }
  .stat-val { font-size: 1.7rem; font-weight: 900; color: #ff6d00 !important; }
  .stat-lbl { font-size: 0.72rem; color: #555 !important; margin-top: 2px; }
  .input-card { background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.5rem; }
  .input-card h4 { font-size: 0.9rem; font-weight: 700; color: #ff6d00 !important; margin: 0 0 0.2rem; }
  .input-card small { color: #444 !important; font-size: 0.76rem; }
  .result-card { background: #111; border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 0.5rem; border: 1px solid #1e1e1e; }
  .result-name { font-size: 1.6rem; font-weight: 900; margin: 0 0 0.3rem; }
  .sev-pill { display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; border: 1px solid; }
  .conf-bar-label { font-size: 0.8rem; color: #777 !important; }
  .info-block { background: #0d0d0d; border: 1px solid #1a1a1a; border-left: 3px solid #ff6d00; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #aaa !important; line-height: 1.8; white-space: pre-line; margin-top: 0.4rem; }
  .stTabs [data-baseweb="tab-list"] { background: #0f0f0f; border-radius: 10px; padding: 4px; gap: 6px; border: 1px solid #1a1a1a; margin-bottom: 1rem; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #555 !important; font-weight: 600; font-size: 0.85rem; padding: 6px 16px; }
  .stTabs [aria-selected="true"] { background: #ff6d00 !important; color: #fff !important; }
  hr { border-color: #1a1a1a !important; }
  .warn-box { background: #1a1000; border: 1px solid #ff6d0044; border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 0.82rem; color: #ffab40 !important; margin-top: 0.6rem; }
  .footer { text-align: center; font-size: 0.73rem; color: #2a2a2a !important; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #1a1a1a; }
  .footer span { color: #ff6d00 !important; }
  .guide-card { background: #111; border-radius: 12px; border: 1px solid #1a1a1a; padding: 1rem; text-align: center; }
  .stProgress > div > div { background: linear-gradient(90deg, #ff6d00, #ffab40) !important; border-radius: 4px; }
  .stProgress > div { background: #1a1a1a !important; border-radius: 4px; }
</style>
"""

# Inject custom styles into the Streamlit page.
st.markdown(CSS, unsafe_allow_html=True)


# ── Model utilities ───────────────────────────────────────────────────────────

def build_model():
    """
    Reconstruct the EfficientNetB0 classification architecture used during training.

    Architecture:
        Input (224×224×3)
        → EfficientNetB0 base (frozen, no top)
        → GlobalAveragePooling2D
        → Dropout(0.5)
        → Dense(128, ReLU)
        → Dropout(0.3)
        → Dense(4, Softmax)   ← one output per disease class

    weights=None because we load the saved weights separately via load_model().
    training=False keeps BatchNorm layers in inference mode at all times.

    Returns:
        keras.Model: Uncompiled model ready to receive loaded weights.
    """
    base = EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
    base.trainable = False                       # Freeze base — weights loaded from file
    inp = keras.Input(shape=(224, 224, 3))
    x   = base(inp, training=False)              # training=False → BatchNorm in inference mode
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(0.5)(x)                # Dropout(0.5) after pooling (matches training)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)                # Dropout(0.3) before output layer
    out = layers.Dense(4, activation="softmax")(x)  # 4 output nodes for 4 disease classes
    return models.Model(inp, out)


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Build the model and load pre-trained weights from disk.

    Uses st.cache_resource so the model is initialised only once per session,
    avoiding repeated disk reads on every page interaction.

    Returns:
        keras.Model | None: Loaded model, or None if the weights file is missing.
    """
    m = build_model()
    if not os.path.exists(WEIGHTS_FILE):
        # Weights file not found — return None so the UI can show a helpful message.
        return None
    m.load_weights(WEIGHTS_FILE)
    return m


def preprocess_image(pil_image):
    """
    Prepare a PIL image for model inference.

    Steps:
        1. Convert to RGB (handles RGBA / grayscale inputs safely).
        2. Resize to 224×224 px (model input size).
        3. Cast pixels to float32.
        4. Apply EfficientNet preprocessing (scales pixel values to [-1, 1]).
        5. Add batch dimension → shape (1, 224, 224, 3).

    Args:
        pil_image (PIL.Image.Image): Raw image from file upload or camera.

    Returns:
        np.ndarray: Preprocessed array of shape (1, 224, 224, 3).
    """
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return preprocess_input(np.expand_dims(arr, 0))  # scales pixels to [-1, 1]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sb-title">🌽 MaizeAI</p>', unsafe_allow_html=True)
    st.markdown("<small style='color:#444'>Crop Disease Detection System</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**About This Project**")
    st.markdown(
        "<small style='color:#666'>This system uses EfficientNetB0 deep learning to "
        "classify maize leaf diseases from a single photograph. "
        "Built for TechCrush AI/ML Bootcamp Cohort 5.</small>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Disease Classes**")
    for cls, inf in DISEASE_INFO.items():
        st.markdown(
            f'<div class="sb-card">{inf["emoji"]} <b>{cls}</b><br>'
            f'<span style="color:{inf["severity_color"]};font-size:0.75rem">'
            f'Severity: {inf["severity"]}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Model Details**")
    for k, v in [
        ("Architecture", "EfficientNetB0"),
        ("Strategy", "Transfer Learning"),
        ("Input size", "224 × 224 px"),
        ("Test accuracy", "91.69%"),
        ("Macro F1", "89.46%"),
        ("Classes", "4"),
    ]:
        st.markdown(f'<div class="sb-card"><b>{k}:</b> {v}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#2a2a2a'>TechCrush Cohort 5 · Capstone Project · "
        "Built for academic ML research.</small>",
        unsafe_allow_html=True,
    )


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">🌽 AI-Powered Agriculture</div>
  <h1>Maize Disease<br><span>Detection System</span></h1>
  <p>Upload a maize leaf photo or use your camera to get an instant AI diagnosis,
  confidence scores, and treatment advice.</p>
</div>
""", unsafe_allow_html=True)

# Stat bar
s1, s2, s3, s4 = st.columns(4)
for col, val, lbl in [
    (s1, "91.69%", "Test Accuracy"),
    (s2, "89.46%", "Macro F1 Score"),
    (s3, "4", "Disease Classes"),
    (s4, "224 px", "Input Resolution"),
]:
    col.markdown(
        f'<div class="stat-box"><div class="stat-val">{val}</div>'
        f'<div class="stat-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Diagnose", "📖 Disease Guide", "📊 Performance"])


# ── TAB 1: DIAGNOSE ───────────────────────────────────────────────────────────
with tab1:
    col_cam, col_up = st.columns(2, gap="medium")

    with col_cam:
        st.markdown(
            '<div class="input-card"><h4>📷 Take a Photo</h4>'
            '<small>Works on phones and laptop webcams</small></div>',
            unsafe_allow_html=True,
        )
        camera_photo = st.camera_input("Camera", label_visibility="collapsed")

    with col_up:
        st.markdown(
            '<div class="input-card"><h4>📂 Upload an Image</h4>'
            '<small>Select a saved JPG, JPEG or PNG from your device</small></div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

    image_source = camera_photo if camera_photo is not None else uploaded

    if image_source is not None:
        pil_image = Image.open(io.BytesIO(image_source.read()))

        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.image(pil_image, caption="Input image", use_column_width=True)
            st.markdown(
                f"<small style='color:#333'>Size: {pil_image.width}×{pil_image.height} px</small>",
                unsafe_allow_html=True,
            )

        with col_res:
            with st.spinner("Loading model…"):
                model = load_model()

            if model is None:
                st.error(
                    f"Model weights not found. "
                    f"Place `{WEIGHTS_FILE}` in the same folder as `app.py` and restart."
                )
            else:
                # ── Inference ─────────────────────────────────────────────────
                img_arr    = preprocess_image(pil_image)               # → (1, 224, 224, 3)
                preds      = model.predict(img_arr, verbose=0)[0]      # → (4,) softmax outputs
                pred_idx   = int(np.argmax(preds))                     # Index of highest probability
                raw_class  = CLASS_NAMES[pred_idx]                     # e.g. "Gray_leaf_spot"
                pred_class = CLASS_TO_INFO[raw_class]                  # e.g. "Gray Leaf Spot"
                conf       = float(preds[pred_idx]) * 100              # Confidence as percentage
                info       = DISEASE_INFO[pred_class]                  # Retrieve disease knowledge

                st.markdown(
                    f'<div class="result-card" style="box-shadow:{info["glow"]};'
                    f'border-color:{info["color"]}22">'
                    f'<div class="result-name" style="color:{info["color"]}">'
                    f'{info["emoji"]} {pred_class}</div>'
                    f'<div><b style="font-size:1.2rem;color:#fff">{conf:.1f}%</b>'
                    f'<span style="color:#444;font-size:0.82rem"> confidence</span>'
                    f'&nbsp;<span class="sev-pill" style="color:{info["severity_color"]};'
                    f'border-color:{info["severity_color"]}44;'
                    f'background:{info["severity_color"]}11">'
                    f'{info["severity"]} severity</span></div></div>',
                    unsafe_allow_html=True,
                )

                if conf < CONF_WARN:
                    st.markdown(
                        '<div class="warn-box">⚠️ Low confidence — '
                        'try a clearer, better-lit photo or consult an agronomist.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>**Confidence by class**", unsafe_allow_html=True)
                for i, cls in enumerate(CLASS_NAMES):
                    pct  = float(preds[i]) * 100
                    disp = CLASS_TO_INFO[cls]
                    c1, c2 = st.columns([1, 3])
                    c1.markdown(
                        f"<span class='conf-bar-label'>"
                        f"{DISEASE_INFO[disp]['emoji']} {disp}</span>",
                        unsafe_allow_html=True,
                    )
                    c2.progress(pct / 100, text=f"{pct:.1f}%")

                st.markdown("---")
                # Heading and content block are merged into one HTML string so that
                # Streamlit does not add an extra paragraph gap between them.
                st.markdown(
                    f'<p style="font-weight:700;margin:0 0 0.4rem 0;">Recommended Treatment</p>'
                    f'<div class="info-block">{info["treatment"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p style="font-weight:700;margin:0.8rem 0 0.4rem 0;">Prevention</p>'
                    f'<div class="info-block">{info["prevention"]}</div>',
                    unsafe_allow_html=True,
                )

    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### What this system can detect")
        cols = st.columns(4)
        for col, (cls, inf) in zip(cols, DISEASE_INFO.items()):
            with col:
                st.markdown(
                    f'<div class="guide-card">'
                    f'<div style="font-size:2rem">{inf["emoji"]}</div>'
                    f'<b style="color:#fff">{cls}</b><br>'
                    f'<small style="color:{inf["severity_color"]}">{inf["severity"]} severity</small>'
                    f'<br><br><small style="color:#444;font-size:0.73rem">'
                    f'{inf["description"][:65]}…</small></div>',
                    unsafe_allow_html=True,
                )


# ── TAB 2: DISEASE GUIDE ─────────────────────────────────────────────────────
with tab2:
    st.markdown("## Disease Reference Guide")
    st.markdown("Detailed information on each disease class this model was trained to identify.")
    st.markdown("---")
    for cls, inf in DISEASE_INFO.items():
        with st.expander(f"{inf['emoji']}  {cls}  —  {inf['severity']} severity"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(
                    '<p style="font-weight:700;margin:0 0 0.4rem 0;">Description</p>'
                    f'<div class="info-block">{inf["description"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p style="font-weight:700;margin:0.8rem 0 0.4rem 0;">Cause</p>'
                    f'<div class="info-block">{inf["cause"]}</div>',
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    '<p style="font-weight:700;margin:0 0 0.4rem 0;">Treatment</p>'
                    f'<div class="info-block">{inf["treatment"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p style="font-weight:700;margin:0.8rem 0 0.4rem 0;">Prevention</p>'
                    f'<div class="info-block">{inf["prevention"]}</div>',
                    unsafe_allow_html=True,
                )


# ── TAB 3: PERFORMANCE ────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Model Performance Results")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Model Comparison")
        st.table({
            "Model":    ["Baseline DNN", "EfficientNetB0 (This Project)"],
            "Accuracy": ["82.82%",        "91.69%"],
            "Loss":     ["0.4143",        "0.2565"],
            "Macro F1": ["—",             "89.46%"],
        })
    with c2:
        st.markdown("### Per-Class Results")
        st.table({
            "Class":     ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"],
            "Precision": ["0.80",   "0.99",        "0.87",          "0.99"],
            "Recall":    ["0.96",   "0.92",        "0.67",          "1.00"],
            "F1 Score":  ["0.87",   "0.95",        "0.76",          "1.00"],
        })

    st.markdown("""
---
### Training Strategy

**Phase 1 — Head Training**
EfficientNetB0 base weights were frozen. Only the classification head (GlobalAveragePooling → Dense 128 → Dense 4) was trained for up to 25 epochs with a learning rate of 3×10⁻⁴. Early stopping prevented overfitting.

**Phase 2 — Fine-Tuning**
The top 100 layers of EfficientNetB0 were unfrozen and trained at a much lower learning rate of 5×10⁻⁶ for up to 10 epochs. This allowed the model to adapt pre-learned ImageNet features to maize disease patterns.

**Techniques used:**
- Data augmentation (rotation, zoom, flip, brightness)
- Class balancing with computed class weights
- Early stopping and ReduceLROnPlateau callbacks

### Dataset
PlantVillage Maize Disease Dataset (Kaggle) · 4 classes · 70 / 20 / 10 split · 224×224 px input
    """)


st.markdown(
    '<div class="footer">🌽 Maize Disease Detection System &nbsp;|&nbsp; '
    '<span>TechCrush AI/ML Bootcamp Cohort 5</span> &nbsp;|&nbsp; '
    'EfficientNetB0 · Streamlit · Transfer Learning</div>',
    unsafe_allow_html=True,
)
