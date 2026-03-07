import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from preprocess import preprocess_image

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Handwritten Character Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# LOAD CSS
# ===============================
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
    <div class="app-header">
        <h1>Handwritten Character Recognition</h1>
        <p>Neural Analysis System &nbsp;·&nbsp; CNN Model</p>
    </div>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & LABELS
# ===============================
@st.cache_resource
def load_assets():
    model = load_model("handwritten_cnn_model (2).keras")
    class_names = np.load("class_names (2).npy", allow_pickle=True)
    return model, class_names

model, class_names = load_assets()
st.success(f"Model loaded — {len(class_names)} classes ready")

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_and_display(img_pil, true_label=None, show_input=True):
    if show_input:
        col_img, col_spacer = st.columns([1, 3])
        with col_img:
            st.image(img_pil, width=200)

    img_array = preprocess_image(img_pil)
    if img_array is None:
        st.warning("No visible strokes detected.")
        return

    predictions = model.predict(img_array)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # ── Result Card ──
    status_color = "#4a9e6b" if confidence >= 50 else "#c5884e"
    status_text  = "HIGH CONFIDENCE" if confidence >= 80 else "MODERATE CONFIDENCE" if confidence >= 50 else "LOW CONFIDENCE"

    st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">{predicted_label}</div>
            <div class="prediction-confidence">
                <span style="color:{status_color};">{status_text}</span>
                &nbsp;&nbsp;{confidence:.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # ── Top 3 Predictions ──
    with col_left:
        st.markdown("""
            <div style="margin-bottom:0.75rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                letter-spacing:0.25em;text-transform:uppercase;color:#c5a84e;">
                    Top Predictions
                </span>
            </div>
        """, unsafe_allow_html=True)

        top_3 = predictions.argsort()[-3:][::-1]
        rows_html = ""
        for rank, idx in enumerate(top_3):
            pct = predictions[idx] * 100
            bar_width = int(pct)
            opacity = 1.0 - rank * 0.2
            rows_html += f"""
                <div class="top-prediction-row" style="opacity:{opacity}">
                    <span class="top-prediction-char">{class_names[idx]}</span>
                    <div class="top-prediction-bar">
                        <div class="top-prediction-fill" style="width:{bar_width}%"></div>
                    </div>
                    <span class="top-prediction-pct">{pct:.1f}%</span>
                </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)

    # ── Metrics ──
    with col_right:
        st.markdown("""
            <div style="margin-bottom:0.75rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                letter-spacing:0.25em;text-transform:uppercase;color:#c5a84e;">
                    Metrics
                </span>
            </div>
        """, unsafe_allow_html=True)

        if true_label is not None:
            match = predicted_label == true_label
            acc = prec = rec = f1 = 100.0 if match else 0.0
            metrics_html = f"""
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">{acc:.0f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value">{prec:.0f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value">{rec:.0f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">F1 Score</div>
                        <div class="metric-value">{f1:.0f}%</div>
                    </div>
                </div>
            """
        else:
            metrics_html = f"""
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{confidence:.1f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Class Index</div>
                        <div class="metric-value">{predicted_index}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Total Classes</div>
                        <div class="metric-value">{len(class_names)}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Rank</div>
                        <div class="metric-value">#1</div>
                    </div>
                </div>
            """
        st.markdown(metrics_html, unsafe_allow_html=True)

# ===============================
# TWO COLUMN LAYOUT
# ===============================
st.markdown("<br>", unsafe_allow_html=True)
col_upload, col_canvas = st.columns(2, gap="large")

# ── Upload Image ──
with col_upload:
    st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
        letter-spacing:0.25em;text-transform:uppercase;color:#c5a84e;margin-bottom:0.75rem;">
            Upload Image
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        predict_and_display(img_pil)

# ── Draw Canvas ──
with col_canvas:
    st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
        letter-spacing:0.25em;text-transform:uppercase;color:#c5a84e;margin-bottom:0.75rem;">
            Draw a Character
        </div>
        <p style="font-size:0.75rem !important;color:#5a5850 !important;
        margin-bottom:0.75rem !important;">
            Black stroke on white background
        </p>
    """, unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        img_pil = Image.fromarray(img_array.astype("uint8"), "RGBA").convert("RGB")
        gray = np.mean(np.array(img_pil.convert("L")))

        if gray > 250:
            st.info("Draw something on the canvas above to analyze")
        else:
            predict_and_display(img_pil, show_input=False)

# ── Footer ──
st.markdown("""
    <div style="
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #2a2d3a;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.2em;
        color: #3a3830;
        text-transform: uppercase;
    ">
        Handwritten Character Recognition &nbsp;·&nbsp; CNN Model &nbsp;·&nbsp; Neural Analysis System
    </div>
""", unsafe_allow_html=True)