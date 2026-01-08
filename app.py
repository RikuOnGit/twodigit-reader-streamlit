import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Detection helpers
# -----------------------------
def detect_digit_boxes(gray_img):
    """
    gray_img: 2D uint8 image (H,W), values 0..255
    returns: (boxes, th)
      boxes: list of (x,y,w,h) sorted left->right
      th: threshold image used for contour detection
    """
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    H, W = gray_img.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Filter tiny specks
        if w * h < 50:
            continue

        # Filter boxes that are too thin/flat
        if h < 10 or w < 5:
            continue

        # Clamp bounds
        x = max(0, x)
        y = max(0, y)
        w = min(W - x, w)
        h = min(H - y, h)

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes, th


def crop_and_prepare(gray_img, box, out_size=28, pad=8, invert=True):
    """
    Crop a detected box and convert to model input: (28,28,1) float in [0,1]
    """
    x, y, w, h = box
    H, W = gray_img.shape

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    crop = gray_img[y0:y1, x0:x1]

    # MNIST-style: bright digit on dark background
    if invert:
        crop = 255 - crop

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=-1)  # (28,28,1)
    return crop


def annotate_boxes(gray_img, boxes):
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return vis


def rgba_to_gray_uint8(img_rgba):
    """
    st_canvas returns image_data often as float array (0..1 or 0..255).
    Convert robustly to uint8 0..255 grayscale.
    """
    arr = np.array(img_rgba)
    if arr.dtype != np.uint8:
        # If 0..1 floats
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # Ensure it has 4 channels (RGBA)
    if arr.ndim == 3 and arr.shape[2] == 4:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        # already grayscale
        gray = arr

    return gray


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Digit Reader", layout="wide")
st.title("Digit reader (detect + classify)")
st.write("Choose an input method, then click **Predict**. The app detects digit boxes and classifies each digit using your trained CNN.")

@st.cache_resource
def load_model():
    return keras.models.load_model("digit_cnn.keras")

model = load_model()

with st.sidebar:
    st.header("Settings")
    pad = st.slider("Padding around detected box", 0, 30, 8)
    invert = st.checkbox("Invert colors before classification", value=True)
    show_debug = st.checkbox("Show debug images", value=True)
    st.caption("If predictions look wrong, try toggling invert.")

mode = st.radio("Input method", ["Draw on canvas", "Upload image"], horizontal=True)

gray = None  # will be set depending on mode

if mode == "Draw on canvas":
    st.caption("Tip: draw the first digit on the left half, second on the right half.")

    canvas_result = st_canvas(
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=560,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is None:
        st.info("Draw something on the canvas.")
        st.stop()

    gray = rgba_to_gray_uint8(canvas_result.image_data)

    do_predict = st.button("Predict from canvas")

else:
    uploaded = st.file_uploader("Upload an image (png/jpg)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()

    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode the uploaded image.")
        st.stop()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    do_predict = st.button("Predict from uploaded image")

# Only run detection/classification after user clicks Predict
if not do_predict:
    st.stop()

if gray is None:
    st.error("No image available.")
    st.stop()

# Detect
boxes, th = detect_digit_boxes(gray)

colA, colB = st.columns(2)

with colA:
    st.subheader("Input + detected boxes")
    if boxes:
        vis = annotate_boxes(gray, boxes)
        st.image(vis, caption=f"Detected {len(boxes)} digit(s)", use_container_width=True)
    else:
        st.image(gray, caption="No boxes detected", use_container_width=True)

with colB:
    if show_debug:
        st.subheader("Threshold debug view")
        st.image(th, caption="Binary image used for contour detection", use_container_width=True)

st.write("Detected boxes:", boxes)

if not boxes:
    st.warning("No digits detected. Try drawing thicker digits, increasing spacing, or using a clearer image.")
    st.stop()

# Crop + predict
crops = np.stack([crop_and_prepare(gray, b, pad=pad, invert=invert) for b in boxes], axis=0)

if show_debug:
    st.write("Crops batch shape:", crops.shape, "min/max:", float(crops.min()), float(crops.max()))

probs = model.predict(crops, verbose=0)
preds = np.argmax(probs, axis=1)

st.subheader("Cropped digits (what the model sees)")
crop_cols = st.columns(min(len(boxes), 6))
for i, (c, p) in enumerate(zip(crops, preds)):
    with crop_cols[i % len(crop_cols)]:
        st.image(c.squeeze(), caption=f"Pred: {p}", use_container_width=True)

number_str = "".join(map(str, preds))
st.success(f"Predicted digits: {list(preds)}  →  **{number_str}**")
