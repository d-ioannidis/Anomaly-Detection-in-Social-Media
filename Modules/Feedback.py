import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class Feedback:
    def __init__(self, fact_checker, anomaly_detector, data_path, preprocessor):
        """
        Initialize the Feedback class for enhanced model improvement.

        Parameters
        ----------
        fact_checker : object
            Instance of a fact-checking module.
        anomaly_detector : AnomalyDetector
            Instance for detecting anomalies in the data.
        data_path : str
            File path to the main dataset CSV file.
        preprocessor : DataPreprocessor
            Instance of data preprocessor to preprocess data.
        """
        self.preprocessor = preprocessor
        self.fact_checker = fact_checker
        self.anomaly_detector = anomaly_detector
        self.data_path = data_path

    def update_system(self):
        # Retrieve the fact-check results
        fact_check_df = pd.read_csv('fact_check_results.csv')

        # Map the results to binary labels
        results_mapping = {
            'False': 1,
            'Partially False': 1,
            'True': 0,
            'Partially True': 0,
            'Unclear': 0
        }

        fact_check_df['label'] = fact_check_df['Fact_Check_Prediction'].map(results_mapping)

        # Retrieve features for the verified tweets
        verified_tweets = fact_check_df['Original Tweets'].tolist()
        main_data = self.preprocessor.data
        mask = main_data['Original Tweets'].isin(verified_tweets)
        verified_indices = main_data[mask].index

        # Prepare the feature and label pairs
        training_data = {}
        for feature_name, feature_list in self.anomaly_detector.suspect_features.items():
            features = np.vstack(feature_list)[verified_indices]
            labels = fact_check_df['label'].values
            training_data[feature_name] = (features, labels)

        # Update the models
        self.update_hybrid_model(training_data)
        self.update_autoencoder(training_data)

        return "Models updated with fact-check results"
    
    def update_hybrid_model(self, training_data):
        for feature_name, (features, labels) in training_data.items():
            # Update each classifier within the hybrid model
            self.anomaly_detector.dt.partial_fit(features, labels, classes=[0, 1])
            self.anomaly_detector.svm.partial_fit(features, labels, classes=[0, 1])
            self.anomaly_detector.nb.partial_fit(features, labels, classes=[0, 1])

    def update_autoencoder(self, training_data):
        for feature_name, (features, labels) in training_data.items():
            normal_data = features[labels == 0]
            if len(normal_data) > 0:
                self.anomaly_detector.autoencoder.fit(
                    normal_data,
                    epochs=20,
                    batch_size=32,
                    verbose=1
                )

    def load_fact_check_results(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(base_dir, '..', 'fact_check_results.csv')
        return pd.read_csv(out_path)