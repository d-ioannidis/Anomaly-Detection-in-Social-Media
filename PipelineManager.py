import sys
sys.path.insert(0, 'Modules')
import os
from Data_Collector import DataCollector
from Data_Preprocessor import DataPreprocessor
from Anomaly_Detection import AnomalyDetector
from Fact_Checker import FactChecker
from Feedback import Feedback
import numpy as np
import pandas as pd

class PipelineManager:
    def __init__(self, config):
        # e.g. file paths, model settings, API keys
        self.collector = DataCollector(config["data_source"])
        self.preprocessor = DataPreprocessor(self.collector)
        self.anomaly_detector = AnomalyDetector()
        self.fact_checker = FactChecker()

    def run(self):
        #1. Collect
        self.collector.load_data()
        #2. Preprocess
        self.preprocessor.preprocess_data()
        embeddings = self.preprocessor.bert_tokenize()
        dtm = self.preprocessor.calculate_document_term_matrix()
        tfidf = self.preprocessor.calculate_term_frequency_inverse_document_frequency_matrix()

        # Access and display sentiment and emotion scores from the preprocessed data
        sentiment_labels = self.preprocessor.data['Sentiment_Label'].values
        sentiment_scores = self.preprocessor.data['Sentiment_Score'].values
        emotion_scores = self.preprocessor.data['Emotion_Labels'].values
        
        print("Sentiment Scores:", sentiment_scores)
        print("Sentiment Labels:", sentiment_labels)
        print("Emotion Scores:", emotion_scores)

        #3. Anomaly Detection
        anomaly_results = {
            'kmeans_tfidf': self.anomaly_detector.kmeans_anomaly_detection(tfidf),
            'kmeans_dtm': self.anomaly_detector.kmeans_anomaly_detection(dtm),
            'kmeans_bert': self.anomaly_detector.kmeans_anomaly_detection(embeddings),
            'dbscan_tfidf': self.anomaly_detector.dbscan_anomaly_detection(tfidf),
            'dbscan_dtm': self.anomaly_detector.dbscan_anomaly_detection(dtm),
            'dbscan_bert': self.anomaly_detector.dbscan_anomaly_detection(embeddings),
            'hybrid_dtsvmnb_tfidf': self.anomaly_detector.hybrid_dtsvmnb_anomaly_detection(tfidf, self.preprocessor.data['Disaster']),
            'hybrid_dtsvmnb_dtm': self.anomaly_detector.hybrid_dtsvmnb_anomaly_detection(dtm, self.preprocessor.data['Disaster']),
            #'hybrid_dtsvmnb_bert': anomaly_detector.hybrid_dtsvmnb_anomaly_detection(bert_embeds, data_preprocessor.data['Disaster']),
            'autoencoder_tfidf': self.anomaly_detector.autoencoder_anomaly_detection(tfidf),
            'autoencoder_dtm': self.anomaly_detector.autoencoder_anomaly_detection(dtm),
            'autoencoder_bert': self.anomaly_detector.autoencoder_anomaly_detection(embeddings)
        }

        anomaly_results_list = list(anomaly_results.values())

        # union of anomalies
        suspect_idx = []

        feature_types = {
            'tfidf': tfidf,
            'dtm': dtm,
            'bert': embeddings
        }

        for arr in anomaly_results_list:
            if isinstance(arr, np.ndarray) and arr.dtype == bool:
                suspect_idx.extend(np.where(arr)[0])
            elif isinstance(arr, pd.Series):
                suspect_idx.extend(arr[arr].index.tolist())
            else:
                print(f"Warning: Anomaly detection result not a boolean array/Series: {type(arr)}")

        suspect_idx = list(set(suspect_idx))

        valid_suspect_idx = [idx for idx in suspect_idx if idx < len(self.preprocessor.data)]
        suspects = self.preprocessor.data.iloc[valid_suspect_idx]

        for name, matrix in feature_types.items():
            if hasattr(matrix, 'tocoo'):
                matrix_for_caching = matrix.toarray()
            else: 
                matrix_for_caching = matrix
            
            if isinstance(matrix_for_caching, np.ndarray):
                suspect_features = matrix_for_caching[valid_suspect_idx]
            else:
                suspect_features = matrix_for_caching.iloc[valid_suspect_idx]

            self.anomaly_detector.cache_suspect_features(
                name, 
                suspect_features, 
                suspect_idx
            )

        #4. Fact-check auto
        if not os.path.isfile('fact_check_results.csv'):
            verification = self.fact_checker.fact_check_tweets(suspects)
        else:
            print("Skipping automatic fact-checking: 'fact_check_results.csv' already exists.")
            verification = pd.read_csv('fact_check_results.csv')

        #5. Feedback
        feedback = Feedback(
            self.fact_checker, 
            self.anomaly_detector, 
            self.collector.data_source,
            self.preprocessor
        )

        if 'verification' in locals():
            return verification
        else:
            return {}
