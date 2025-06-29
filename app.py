import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import model as clf_model
import time


# Load model and tokenizer
@st.cache_resource
def load_model():
    checkpoint = torch.load("/model/checkpoints", map_location=torch.device("cpu"))
    model = clf_model.BERTClassifier(num_classes=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


model, tokenizer = load_model()

# Class names and their corresponding GIFs/images
class_info = {
    0: {
        "name": "World",
        "gif": "https://tenor.com/view/doge-dedcel-dogecoin-dogecoinrise-world-gif-21404969",
        "description": "Global news and international affairs",
    },
    1: {
        "name": "Sports",
        "gif": "https://tenor.com/view/kickball-strike-out-sports-fail-fall-oops-gif-16174448",
        "description": "Sports news and athletic competitions",
    },
    2: {
        "name": "Business",
        "gif": "https://tenor.com/view/ramdev-baba-to-indian-government-pawpaw-business-money-cash-gif-18905018",
        "description": "Business and financial news",
    },
    3: {
        "name": "Sci/Tech",
        "gif": "https://tenor.com/view/anchorman-science-its-science-gif-20674302",
        "description": "Science and technology developments",
    },
}

# Custom CSS for styling
st.markdown(
    """
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #3498db;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: white;
    }
    .class-name {
        font-size: 1.8em;
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence {
        font-size: 1.2em;
        color: #27ae60;
    }
    .description {
        font-style: italic;
        color: #7f8c8d;
    }
    .gif-container {
        display: flex;
        justify-content: center;
        margin: 15px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App title and description
st.markdown(
    '<div class="title">🌍📰 News Article Classifier</div>', unsafe_allow_html=True
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


# Display results in a creative way
if st.button("Classify Article") and text:
    with st.spinner("Analyzing the article..."):
        # Add a small delay to show the spinner
        time.sleep(1)

        predicted_class, probs = predict(text)
        class_data = class_info[predicted_class]

        # Display prediction with animation
        st.balloons()

        # Create a card for the prediction
        st.markdown(
            f"""
        <div class="prediction-card">
            <div class="class-name">🏆 Predicted Category: {class_data['name']}</div>
            <div class="confidence">🔍 Confidence: {probs[predicted_class]*100:.1f}%</div>
            <div class="description">📝 {class_data['description']}</div>
            <div class="gif-container">
                <img src="{class_data['gif']}" width="300">
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Show confidence for all classes
        st.subheader("Confidence Scores for All Categories:")

        cols = st.columns(4)
        for i, (class_id, data) in enumerate(class_info.items()):
            with cols[i]:
                # Create a progress bar for each class
                st.metric(label=data["name"], value=f"{probs[class_id]*100:.1f}%")
                st.progress(probs[class_id])

                # Small icon for each category
                icons = ["🌍", "⚽", "💼", "🔬"]
                st.write(f"{icons[i]} {data['description']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d;">
        <p>Built with 🤖 BERT model and ❤️ Streamlit</p>
        <p>Classifies news articles into World, Sports, Business, or Sci/Tech categories</p>
    </div>
""",
    unsafe_allow_html=True,
)
