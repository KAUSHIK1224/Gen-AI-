import gradio as gr
from transformers import pipeline

# Load Hugging Face's sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    
    if label == "POSITIVE":
        return "😊 Positive"
    elif label == "NEGATIVE":
        return "😠 Negative"
    else:
        return "😐 Neutral"

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis",
    description="Enter text to analyze sentiment (Positive, Negative)."
)

# Launch the app
iface.launch()
