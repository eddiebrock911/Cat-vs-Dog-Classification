import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import gdown

# https://drive.google.com/file/d/1hXaREPSFb0bm-4HHQAWOhNm-oEXTW-1n/view?usp=drive_link

# streamlit run cat-vs-dog.py

# TensorFlow import with graceful fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except ImportError:
    tf = None
    keras_load_model = None
    TF_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PawVision AI | Cat vs Dog",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.15), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(236,72,153,0.12), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(14,165,233,0.10), transparent 35%),
        #070b14;
    color: #f8fafc;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,15,28,0.98), rgba(5,9,18,0.98));
    border-right: 1px solid rgba(148,163,184,0.10);
}
.sidebar-title { text-align: center; padding: 15px 5px 25px 5px; }
.sidebar-logo { font-size: 55px; margin-bottom: 5px; }
.sidebar-name { font-size: 21px; font-weight: 800; color: #ffffff; }
.sidebar-sub { color: #64748b; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; margin-top: 5px; }
.hero { padding: 35px 20px 25px 20px; text-align: center; }
.hero-badge {
    display: inline-block; padding: 7px 15px; border-radius: 30px;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(129,140,248,0.25);
    color: #a5b4fc; font-size: 12px; font-weight: 700; letter-spacing: 1px; margin-bottom: 18px;
}
.hero h1 {
    font-size: clamp(38px, 6vw, 68px); line-height: 1.05; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #ffffff, #a5b4fc, #f9a8d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { max-width: 720px; margin: 18px auto 0 auto; color: #94a3b8; font-size: 16px; line-height: 1.7; }
.glass-card {
    background: rgba(15,23,42,0.65); border: 1px solid rgba(148,163,184,0.12);
    border-radius: 22px; padding: 25px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(18px);
}
.card-title { font-size: 18px; font-weight: 700; margin-bottom: 5px; }
.card-subtitle { color: #64748b; font-size: 13px; margin-bottom: 20px; }
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.55); border: 1px dashed rgba(129,140,248,0.45);
    border-radius: 18px; padding: 12px;
}
.stButton > button {
    width: 100%; border: none; border-radius: 13px; padding: 12px 20px; font-weight: 700; color: white;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899); transition: all 0.25s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(99,102,241,0.30); }
.result-card {
    padding: 35px 25px; border-radius: 24px; text-align: center;
    background: linear-gradient(145deg, rgba(30,41,59,0.85), rgba(15,23,42,0.65));
    border: 1px solid rgba(148,163,184,0.15);
}
.result-icon { font-size: 80px; line-height: 1; margin-bottom: 15px; }
.result-label { font-size: 35px; font-weight: 800; margin-bottom: 8px; }
.confidence { font-size: 15px; color: #94a3b8; }
.confidence strong { color: #ffffff; font-size: 22px; }
.metric-card { background: rgba(15,23,42,0.65); border: 1px solid rgba(148,163,184,0.10); border-radius: 17px; padding: 20px; text-align: center; }
.metric-value { font-size: 27px; font-weight: 800; color: #ffffff; }
.metric-label { color: #64748b; font-size: 12px; margin-top: 5px; }
.status { display: flex; align-items: center; gap: 8px; color: #94a3b8; font-size: 13px; }
.status-dot { width: 9px; height: 9px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 12px #22c55e; }
.footer { text-align: center; color: #475569; font-size: 12px; padding: 35px 0 10px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL - NOW USING .keras
# ============================================================

@st.cache_resource
def load_cnn_model():
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

    model_path = "cnn_model.keras"
    model_id = "1hXaREPSFb0bm-4HHQAWOhNm-oEXTW-1n"

    if not os.path.exists(model_path):
        gdown.download(id=model_id, output=model_path, quiet=False)

    model = keras_load_model(model_path)
    return model

try:
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not found")
    model = load_cnn_model()
    model_status = True
except Exception as e:
    model = None
    model_status = False
    model_error = str(e)

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">
        <div class="sidebar-logo">🐾</div>
        <div class="sidebar-name">PawVision AI</div>
        <div class="sidebar-sub">CNN Image Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Model")
    st.markdown(
        f"""
        <div class="status">
            <span class="status-dot" style="background:{'#22c55e' if model_status else '#ef4444'}; box-shadow:0 0 12px {'#22c55e' if model_status else '#ef4444'}"></span>
            {"Model Loaded (.keras)" if model_status else "Model Error"}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.markdown("""
    **Architecture**
    • Conv2D 32 Filters  
    • Conv2D 64 Filters  
    • Conv2D 128 Filters  
    • MaxPooling2D  
    • Flatten  
    • Dense 128  
    • Dense 64  
    • Sigmoid Output
    """)

    st.markdown("---")
    st.markdown("### 🧠 Input")
    st.caption("Expected image size")
    st.code("256 × 256 × 3")
    st.caption("Supported formats")
    st.write("JPG • JPEG • PNG")

    st.markdown("---")
    if st.button("🗑️ Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ DEEP LEARNING VISION SYSTEM (.keras)</div>
    <h1>Cat vs Dog</h1>
    <p>Upload an image and let your trained CNN model determine whether it contains a cat or a dog.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL ERROR
# ============================================================

if not model_status:
    st.error(
        "Model could not be loaded. Make sure `cnn_model.keras` is present in the same directory."
    )
    st.code(model_error)
    if not TF_AVAILABLE:
        st.warning("TensorFlow is not installed. Install it with:\n\n`pip install tensorflow pillow numpy streamlit`")
    st.stop()

# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([1, 1], gap="large")

# ============================================================
# LEFT - UPLOAD
# ============================================================

with left:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📤 Upload Image</div>
        <div class="card-subtitle">Choose a cat or dog image for classification</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption=f"Uploaded: {uploaded_file.name}",
            use_container_width=True
        )

        st.markdown(
            f"""
            <div style="margin-top:15px; padding:15px; border-radius:14px; background:rgba(30,41,59,0.55); border:1px solid rgba(148,163,184,0.10);">
                <b>📐 Dimensions</b>
                <span style="float:right;color:#94a3b8">{image.size[0]} × {image.size[1]}</span>
                <br><br>
                <b>📦 Format</b>
                <span style="float:right;color:#94a3b8">{uploaded_file.type or "Image"}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        predict_button = st.button("🚀 Analyze Image", use_container_width=True)
    else:
        st.info("Upload a JPG, JPEG or PNG image to start prediction.")
        predict_button = False

# ============================================================
# RIGHT - RESULT
# ============================================================

with right:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🔮 AI Prediction</div>
        <div class="card-subtitle">CNN classification result (.keras model)</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file and predict_button:
        # Preprocess
        img = image.resize((256, 256))
        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        with st.spinner("Analyzing image with CNN (.keras)..."):
            start_time = time.time()
            prediction = model.predict(img_array, verbose=0)
            prediction_time = time.time() - start_time

        probability = float(np.squeeze(prediction))

        # 0 = Cat, 1 = Dog
        if probability >= 0.5:
            result = "Dog"
            icon = "🐶"
            confidence = probability * 100
        else:
            result = "Cat"
            icon = "🐱"
            confidence = (1 - probability) * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-icon">{icon}</div>
                <div class="result-label">{result}</div>
                <div class="confidence">Confidence<br><strong>{confidence:.2f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.progress(min(confidence / 100, 1.0), text=f"Confidence: {confidence:.2f}%")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{confidence:.1f}%</div><div class="metric-label">Confidence</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{prediction_time:.2f}s</div><div class="metric-label">Inference Time</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">256²</div><div class="metric-label">Input Size</div></div>""", unsafe_allow_html=True)

        st.write("")
        st.markdown("### 📊 Prediction Probability")
        cat_probability = (1 - probability) * 100
        dog_probability = probability * 100

        st.write(f"🐱 Cat — **{cat_probability:.2f}%**")
        st.progress(cat_probability / 100)
        st.write(f"🐶 Dog — **{dog_probability:.2f}%**")
        st.progress(dog_probability / 100)

        st.session_state.history.insert(
            0,
            {
                "Image": uploaded_file.name,
                "Prediction": result,
                "Confidence": f"{confidence:.2f}%",
                "Time": f"{prediction_time:.2f}s"
            }
        )
        st.session_state.history = st.session_state.history[:10]
    else:
        st.markdown("""
        <div class="result-card">
            <div class="result-icon">🤖</div>
            <div class="result-label">Waiting...</div>
            <div class="confidence">Upload an image and click <b>Analyze Image</b></div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.history:
    st.write("")
    st.write("")
    st.markdown("## 📜 Prediction History")
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    🐾 PawVision AI<br>
    Powered by Convolutional Neural Network (.keras)<br><br>
    Cat vs Dog Image Classification System
</div>
""", unsafe_allow_html=True)
