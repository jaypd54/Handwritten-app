# Handwritten Character Recognition

A deep learning web application for recognizing handwritten characters using a Convolutional Neural Network (CNN). Built with TensorFlow, Streamlit, and Python.

---

## Overview

This app allows users to either upload an image or draw directly on a canvas to have handwritten characters recognized in real time. The model was trained on 64×64 grayscale images and supports multiple character classes.

---

## Features

- Draw characters on an interactive canvas
- Upload image files for recognition
- Real-time CNN inference with confidence scoring
- Top 3 predictions with visual confidence bars
- Dark-themed professional UI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras (CNN) |
| Frontend | Streamlit |
| Image Processing | Pillow, NumPy |
| Canvas Input | streamlit-drawable-canvas |

---

## Getting Started

### Prerequisites

- Python 3.11
- pip

### Installation

1. Clone the repository

```bash
git clone https://github.com/jaypd54/Handwritten-app.git
cd Handwritten-app
```

2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## Project Structure

```
handwritten_cnn/
├── app.py                          # Main Streamlit application
├── preprocess.py                   # Image preprocessing pipeline
├── style.css                       # UI styling
├── handwritten_cnn_model (2).keras # Trained CNN model
├── class_names (2).npy             # Class label array
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Model Details

- **Input shape:** 1 × 64 × 64 × 1 (grayscale)
- **Architecture:** Convolutional Neural Network
- **Normalization:** Pixel values scaled to [0.0, 1.0]
- **Output:** Softmax probabilities over character classes

---

## Usage

**Drawing mode**
- Use the canvas on the right to draw a character with your mouse
- Use black strokes on the white background
- Prediction updates automatically after drawing

**Upload mode**
- Click the upload area on the left
- Supported formats: PNG, JPG, JPEG
- Best results with black strokes on a white background

---

## License

This project is for educational purposes.
