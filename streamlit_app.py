import os
import streamlit as st
import torch
from transformers import AutoTokenizer
import src.model as clf_model
import numpy as np
from PIL import Image
import time


# Load model and tokenizer
@st.cache_resource
def load_model():
    model = clf_model.BERTClassifier(num_classes=4)

    # Use relative path instead of absolute
    checkpoint_path = "/run/media/bigbrother/Study/_Projects/bert-text-classifier/model/bert_clf_model.pt"

    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


model, tokenizer = load_model()

# Class names and their corresponding imgs - using relative paths
class_info = {
    0: {
        "name": "World",
        "img": "/run/media/bigbrother/Study/_Projects/bert-text-classifier/report/mems/world.jpg",
        "description": "Global news and international affairs",
        "icon": "🌍",
    },
    1: {
        "name": "Sports",
        "img": "/run/media/bigbrother/Study/_Projects/bert-text-classifier/report/mems/sports.jpeg",
        "description": "Sports news and athletic competitions",
        "icon": "⚽",
    },
    2: {
        "name": "Business",
        "img": "/run/media/bigbrother/Study/_Projects/bert-text-classifier/report/mems/business.jpg",
        "description": "Business and financial news",
        "icon": "💼",
    },
    3: {
        "name": "Sci/Tech",
        "img": "/run/media/bigbrother/Study/_Projects/bert-text-classifier/report/mems/it's science.jpg",
        "description": "Science and technology developments",
        "icon": "🔬",
    },
}

# Set dark theme
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark theme styling
st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .title {
        color: #FFFFFF;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #4FC3F7;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        background-color: #1E1E1E;
        border-left: 5px solid #4FC3F7;
    }
    .class-name {
        font-size: 1.8em;
        font-weight: bold;
        color: #4FC3F7;
    }
    .confidence {
        font-size: 1.2em;
        color: #81C784;
    }
    .description {
        font-style: italic;
        color: #B0BEC5;
    }
    .img-container {
        display: flex;
        justify-content: center;
        margin: 15px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4FC3F7;
    }
    .stTextArea textarea {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    .stRadio label {
        color: white !important;
    }
    .stFileUploader label {
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App title and description
st.markdown(
    '<div class="title">📰 News Article Classifier</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subheader">Powered by BERT AI Model</div>', unsafe_allow_html=True
)
st.write(
    "This app classifies news articles into one of four categories: World, Sports, Business, or Sci/Tech."
)

# Input options
input_method = st.radio("Choose input method:", ("Enter text", "Upload file"))

text = ""
if input_method == "Enter text":
    text = st.text_area("Enter your news article text:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")


# Prediction function
def predict(text):
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=1).numpy()[0]

    predicted_class = np.argmax(probs)
    return predicted_class, probs


# Display results
if st.button("Classify Article") and text:
    with st.spinner("Analyzing the article..."):
        time.sleep(1)  # Small delay for spinner

        predicted_class, probs = predict(text)
        class_data = class_info[predicted_class]

        # Check and display img
        img_path = class_data["img"]

        st.image(img_path, caption=class_data["name"], width=300)

        # Prediction card
        st.markdown(
            f"""
        <div class="prediction-card">
            <div class="class-name">{class_data['icon']} Predicted Category: {class_data['name']}</div>
            <div class="confidence">Confidence: {probs[predicted_class]*100:.1f}%</div>
            <div class="description">{class_data['description']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Confidence scores
        st.subheader("Confidence Scores for All Categories:")
        cols = st.columns(4)
        for i, (class_id, data) in enumerate(class_info.items()):
            with cols[i]:
                st.metric(label=data["name"], value=f"{probs[class_id]*100:.1f}%")
                st.progress(float(probs[class_id]))
                st.write(f"{data['icon']} {data['description']}")
