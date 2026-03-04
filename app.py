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
    layout="wide"  # full-screen width
)

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Handwritten Character Recognition</h1>
        <p style='font-size:18px;'>Upload an image or draw a character below.<br>
        <b>Black stroke on white background</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

# ===============================
# LOAD MODEL & LABELS
# ===============================
@st.cache_resource
def load_assets():
    model = load_model("handwritten_cnn_model (2).keras")
    class_names = np.load("class_names (2).npy", allow_pickle=True)
    return model, class_names

model, class_names = load_assets()
st.success(f"Model loaded ({len(class_names)} classes ready)")

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_and_display(img_pil, true_label=None, show_input=True):
    # Only show the input image if needed
    if show_input:
        st.image(img_pil, caption="Input Image", width=300)

    img_array = preprocess_image(img_pil)
    if img_array is None:
        st.warning("No visible strokes detected.")
        return

    predictions = model.predict(img_array)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Prediction display
    st.subheader("Prediction")
    if confidence < 50:
        st.warning("Low confidence — unclear input")
    else:
        st.success(f"**{predicted_label}** ({confidence:.2f}%)")

    # Top 3 predictions
    st.subheader("📊 Top 3 Predictions")
    top_3 = predictions.argsort()[-3:][::-1]
    for idx in top_3:
        st.write(f"**{class_names[idx]}** — {predictions[idx]*100:.2f}%")
        st.progress(float(predictions[idx]))

    # Metrics section
    st.subheader("Metrics")
    st.write(f"- Confidence: {confidence:.2f}%")
    if true_label is not None:
        accuracy = 1.0 if predicted_label == true_label else 0.0
        precision = 1.0 if predicted_label == true_label else 0.0
        recall = 1.0 if predicted_label == true_label else 0.0
        f1 = 1.0 if predicted_label == true_label else 0.0
        st.write(f"- Accuracy: {accuracy*100:.2f}%")
        st.write(f"- Precision: {precision*100:.2f}%")
        st.write(f"- Recall: {recall*100:.2f}%")
        st.write(f"- F1 Score: {f1*100:.2f}%")
    else:
        st.write("Metrics based on confidence only (no true label provided)")

# ===============================
# UPLOAD IMAGE
# ===============================
uploaded_file = st.file_uploader(
    "📂 Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    predict_and_display(img_pil)  # show_input=True by default

# ===============================
# DRAW CANVAS
# ===============================
st.subheader("Draw a Character")
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
        st.info("Draw something first 👀")
    else:
        # Do NOT show the input image again when drawing
        predict_and_display(img_pil, show_input=False)
