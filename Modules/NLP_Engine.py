import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class NLPEngine:
    def __init__(self):
        """
        Initializes the NLP Engine.

        This method initializes the NLP Engine by setting the device (either a CUDA GPU or CPU),
        and loading the sentiment analysis and emotion detection models.

        The sentiment analysis model is a multilingual model that supports sentiment analysis
        in multiple languages.

        The emotion detection model is specifically trained to detect emotions in English text.

        The device is set to the first available CUDA GPU if one is available, otherwise it is set
        to the CPU.

        :return: None
        """
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
        """
        Returns the sentiment scores for the given text.

        The sentiment scores are a dictionary of sentiment labels mapped to their respective scores.
        The sentiment labels are 'positive', 'negative', and 'neutral'.
        The scores are a probability between 0 and 1, where 1 is 100% likely to belong to the given label.

        If the text is empty, the method returns a single dictionary with label 'neutral' and score 0.0.

        If there is an error in analyzing the sentiment, the method prints the error and returns a single dictionary with label 'error' and score 0.0.

        :param text: The text to analyze the sentiment of
        :return: The sentiment scores for the given text
        """
        if not text.strip():
            return [{'label': 'neutral', 'score': 0.0}]
        try:
            return self.sentiment_pipeline(text)
        except Exception as e:
            print(f"Error in sentiment analysis for text: {text}. Error: {e}")
            
            return [{'label': 'error', 'score': 0.0}]
        
    def get_emotion_labels(self, text):
        """
        Returns the emotion labels for the given text.

        The emotion labels are a list of dictionaries, where each dictionary contains the emotion label
        and its corresponding score. The emotion labels are one of 'neutral', 'happy', 'sad', 'angry',
        'fear', 'surprise', 'disgust', or 'error' in case of an analysis error.

        The score is a probability between 0 and 1, where 1 is 100% likely to belong to the given label.

        If the text is empty, the method returns a single dictionary with label 'neutral' and score 0.0.

        If there is an error in analyzing the emotion, the method prints the error and returns a single dictionary with label 'error' and score 0.0.

        :param text: The text to analyze the emotion of
        :return: The emotion labels for the given text
        """
        
        if not text.strip():
            
            return [{'label': 'neutral', 'score': 0.0}]
        try:

            return self.emotion_pipeline(text)
        except Exception as e:
            print(f"Error in emotion analysis for text: {text}. Error: {e}")

            return [{'label': 'error', 'score': 0.0}]