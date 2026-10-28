import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import shutil
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Tesseract
pytesseract.pytesseract.tesseract_cmd = (
    shutil.which("tesseract")
    or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.set_page_config(
    page_title="AI Lost & Found Match Assistant",
    page_icon="🔎"
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
    _, threshold = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return pytesseract.image_to_string(threshold).strip()

st.title("AI Lost & Found Match Assistant")

col1, col2 = st.columns(2)

with col1:
    lost_file = st.file_uploader(
        "Upload Lost Notice", type=["png", "jpg", "jpeg"]
    )

with col2:
    found_file = st.file_uploader(
        "Upload Found Notice", type=["png", "jpg", "jpeg"]
    )

if st.button("Find Match") and lost_file and found_file:

    lost_text = extract_text(Image.open(lost_file))
    found_text = extract_text(Image.open(found_file))

    st.subheader("Extracted Text")
    st.write("**Lost:**", lost_text)
    st.write("**Found:**", found_text)

    if lost_text and found_text:
        model = load_model()

        a = model.encode([lost_text])
        b = model.encode([found_text])

        score = cosine_similarity(a, b)[0][0] * 100

        st.metric("Match Percentage", f"{score:.2f}%")

        if score >= 70:
            st.success("Possible Match Found")
        elif score >= 45:
            st.warning("Possible Partial Match")
        else:
            st.error("No Strong Match Found")
