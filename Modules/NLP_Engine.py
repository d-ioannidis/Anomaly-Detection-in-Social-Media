import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class NLPEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"NLP Engine initialized with model: Emotion English DistilRoBERTa-base")

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )

        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            tokenizer="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=0 if torch.cuda.is_available() else -1
        )

    def get_sentiment_scores(self, text):
        if not text.strip():
            return [{'label': 'neutral', 'score': 0.0}]
        try:
            return self.sentiment_pipeline(text)
        except Exception as e:
            print(f"Error in sentiment analysis for text: {text}. Error: {e}")
            return [{'label': 'error', 'score': 0.0}]
        
    def get_emotion_labels(self, text):
        if not text.strip():
            return [{'label': 'neutral', 'score': 0.0}]
        try:
            return self.emotion_pipeline(text)
        except Exception as e:
            print(f"Error in emotion analysis for text: {text}. Error: {e}")
            return [{'label': 'error', 'score': 0.0}]