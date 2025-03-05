import gradio as gr
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})"


custom_css = """
#interface-container { background-color: #eef2f3; } 
#title { color: #1e3a5f; font-size: 24px; } 
"""


iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter Text", placeholder="Type your sentence here..."),
    outputs=gr.Text(label="Sentiment Analysis Result"),
    title="Sentiment Analysis API",
    description=" Enter a sentence, and the model will predict if it's POSITIVE or NEGATIVE.",
    theme="compact",
    css=custom_css
)

iface.launch()
