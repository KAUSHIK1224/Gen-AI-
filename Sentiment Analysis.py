# -*- coding: utf-8 -*-
"""Sentiment Analysis.ipynb
Original file is located at
    https://colab.research.google.com/drive/13twMOaCHkaj306XryICSELQxw3gauL2K
"""



!pip install gradio
import gradio as gr
from transformers import pipeline

# Load sentiment analysis model
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label} (Confidence: {score:.2f})"

# Create Gradio UI
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis API",
    description="Enter a sentence, and the model will analyze its sentiment (positive/negative)."
)

# Run Gradio app
iface.launch()
