import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import shutil

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


def extract_text(image):
   
    image_np = np.array(image)

    
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

   
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

   
    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    text = pytesseract.image_to_string(threshold)

    return text.strip()
