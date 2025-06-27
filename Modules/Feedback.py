import pandas as pd
import numpy as np
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
        """
        Updates the system by integrating fact-check results with anomaly detection features 
        to train machine learning models.

        This method processes the fact-check results and merges them with the corresponding 
        anomaly detection features. It then prepares the data for model training and updates 
        both a hybrid model and an autoencoder with the training data.

        Raises
        ------
        ValueError
            If no verified tweets are found in the main data.

        Returns
        -------
        result : dict
            The result of updating the autoencoder with the training data.
        """

        fact_check_df = pd.read_csv('fact_check_results.csv') # Load the fact-check results

        mapping = { # Map fact-check results to labels
            'False': 1, 
            'Partially False': 1, 
            'True': 0, 
            'Partially True': 0, 
            'Unclear': 0
        }

        fact_check_df['label'] = fact_check_df['Fact_Check_Prediction'].map(mapping) # Map fact-check results to labels
        fact_check_df = fact_check_df.dropna(subset=['label'])
        fact_check_df['label'] = fact_check_df['label'].astype(int)
        
        main = self.preprocessor.data[['Original Tweets']].copy()

        for feature_name, feature_list in self.anomaly_detector.suspect_features.items():
            arr = np.atleast_2d(feature_list[0] if len(feature_list) == 1 else feature_list)
            
            if arr.shape[0] != len(main):
                arr = arr.T
            main[feature_name] = list(arr)
        
        merged = fact_check_df[['Original Tweets', 'label']]\
                    .merge(main, on='Original Tweets', how='inner')
        
        if merged.empty:
            raise ValueError("No verified tweets found in main data.")
        
        training_data = {}
        labels = merged['label'].values
        for feature_name in self.anomaly_detector.suspect_features:
            features = np.vstack(merged[feature_name].values)
            training_data[feature_name] = (features, labels)
            
        hybrid_results = self.update_hybrid_model(training_data) # Update the hybrid model with the training data
        autoencoder_results = self.update_autoencoder(training_data) # Update the autoencoder with the training data

        return {
            'hybrid':   hybrid_results,
            'autoenc':  autoencoder_results
        }
    
    def update_hybrid_model(self, training_data):    
        """
        Update the hybrid model with the training data.

        Parameters
        ----------
        training_data : dict
            A dictionary containing the feature names as keys and a tuple of
            (features, labels) as values.

        Returns
        -------
        results : dict
            A dictionary containing the feature names as keys and a dictionary of
            scores as values. The scores dictionary contains the accuracy score for
            each classifier in the hybrid model (DecisionTreeClassifier, SVC, and
            MultinomialNB).
        """
        results = {}
        for feature_name, (features, labels) in training_data.items():
            # Update each classifier within the hybrid model
            dt_score = self.anomaly_detector.dt.fit(features, labels).score(features, labels)
            svm_score = self.anomaly_detector.svm.fit(features, labels).score(features, labels)
            nb_score = self.anomaly_detector.nb.fit(np.clip(features, 0, None), labels).score(features, labels)

            results[feature_name] = {'dt_score': dt_score, 'svm_score': svm_score, 'nb_score': nb_score}
        
        return results

    def update_autoencoder(self, training_data, autoencoder_encoding_dim=128):
        """
        Update the autoencoder with the training data.

        Parameters
        ----------
        training_data : dict
            A dictionary containing the feature names as keys and a tuple of
            (features, labels) as values.
        autoencoder_encoding_dim : int, optional (default=128)
            The number of dimensions in the encoded representation of the input data.

        Returns
        -------
        anomalies : numpy.ndarray
            A boolean array indicating which samples are anomalies in the input data.
        """

        results = {}
        for feature_name, (features, labels) in training_data.items():
            normal_data = features[labels == 0]
            if len(normal_data) > 0:
                anomalies = self.anomaly_detector.autoencoder_anomaly_detection(
                    data=normal_data,
                    encoding_dim=autoencoder_encoding_dim,
                    force_retrain=False
                )
                results[feature_name] = anomalies

        return results

    def load_fact_check_results(self):
        """
        Loads the fact-check results from a CSV file.

        This function constructs the file path for the 'fact_check_results.csv' file,
        which is located in the parent directory of the current file's directory.
        It then reads the CSV file into a pandas DataFrame and returns it.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the fact-check results.
        """

        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(base_dir, '..', 'fact_check_results.csv')

        return pd.read_csv(out_path)