# AI Lost & Found Match Assistant

AI Lost & Found Match Assistant is an AI-powered application that helps compare Lost and Found item notices. It uses OCR to extract text from uploaded images and text similarity to calculate how closely the Lost and Found details match.

## Live Application

https://ai-lost-and-found-match-assistant-mzvmjuv3rpfr7bnpvuk7vs.streamlit.app/


## Features

- Upload Lost item notice images
- Upload Found item notice images
- Extract text using OCR
- Compare Lost and Found descriptions
- Calculate match percentage
- Display possible match, partial match, or no strong match
- Simple and user-friendly Streamlit interface



## Technologies Used

- Python
- Streamlit
- Tesseract OCR
- OpenCV
- Pillow
- NumPy
- Sentence Transformers
- Scikit-learn

## How It Works

1. The user uploads a Lost item notice.
2. The user uploads a Found item notice.
3. Tesseract OCR extracts the text from both images.
4. The extracted texts are converted into embeddings using `all-MiniLM-L6-v2`.
5. Cosine similarity is used to compare the two texts.
6. The similarity score is displayed as a match percentage.
7. The application provides a possible match result based on the score.

## Match Result

| Match Percentage | Result |
|---|---|
| 70% and above | Possible Match Found |
| 45% – 69% | Possible Partial Match |
| Below 45% | No Strong Match Found |


## Installation

Clone the repository:



Move into the project folder:



Install the required packages:


Run the application:

## Requirements

The project requires Python and the dependencies listed in `requirements.txt`.

Tesseract OCR is required for text extraction.

## Author

**Humairul Jashira M.**

B.Sc. Computer Science with AI
