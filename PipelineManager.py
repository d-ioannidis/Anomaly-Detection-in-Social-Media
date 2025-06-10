import sys
sys.path.insert(0, 'Modules')
import os
from Data_Collector import DataCollector
from Data_Preprocessor import DataPreprocessor
from Anomaly_Detection import AnomalyDetector
from Fact_Checker import FactChecker
from Feedback import Feedback
import numpy as np

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
            suspect_idx.extend(np.where(arr)[0])

        suspects = self.preprocessor.data.iloc[suspect_idx]

        for name, matrix in feature_types.items():
            self.anomaly_detector.cache_suspect_features(name, 
                                                         matrix, 
                                                         suspect_idx
            )

        #4. Fact-check auto
        if not os.path.isfile('fact_check_results.csv'):
            verification = self.fact_checker.fact_check_tweets(suspects)

        #5. Feedback
        feedback = Feedback(
            self.fact_checker, 
            self.anomaly_detector, 
            self.collector.data_source,
            self.preprocessor
        )

        verification = feedback.update_system()