import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Feedback:
    def __init__(self, fact_checker, anomaly_detector, data_path, preprocessor):
        """
        Initialize the Feedback class.

        Parameters
        ----------
        fact_checker : object
            Instance of the fact-checking module.
        anomaly_detector : AnomalyDetector
            Instance of the anomaly detector.
        data_path : str
            Path to the main dataset CSV file.
        preprocessor : DataPreprocessor
            Instance of the data preprocessor.
        """
        self.preprocessor = preprocessor
        self.fact_checker = fact_checker
        self.anomaly_detector = anomaly_detector
        self.data_path = data_path

    def save_thresholds(self, path='adaptive_thresholds.json'):
        """
        Save anomaly detection thresholds to a JSON file.
        """
        with open(path, 'w') as f:
            json.dump(self.anomaly_detector.thresholds, f, indent=2)

    def load_thresholds(self, path='adaptive_thresholds.json'):
        """
        Load anomaly detection thresholds from a JSON file if available.
        """
        if os.path.exists(path):
            with open(path, 'r') as f:
                loaded = json.load(f)
                self.anomaly_detector.thresholds.update(loaded)

    def update_thresholds(self, false_positive_rate, missed_anomaly_rate=None, step=1.0):
        """
        Update anomaly thresholds using weak feedback.

        Higher percentile => fewer anomaly alerts
        Lower percentile => more anomaly alerts
        """
        for model_name in self.anomaly_detector.thresholds:
            current = self.anomaly_detector.thresholds[model_name]

            if false_positive_rate > 0.20:
                current += step
            elif missed_anomaly_rate is not None and missed_anomaly_rate > 0.20:
                current -= step

            self.anomaly_detector.thresholds[model_name] = min(
                self.anomaly_detector.max_percentile,
                max(self.anomaly_detector.min_percentile, current)
            )

    def _map_fact_check_label(self, pred):
        """
        Map fact-check result to weak binary supervision.

        Returns
        -------
        int
            1 = anomaly-like / misinformation-like
            0 = normal / non-anomalous
        """
        pred_str = str(pred).strip().lower()

        if 'false' in pred_str:
            return 1
        elif 'true' in pred_str or 'well supported' in pred_str:
            return 0
        elif 'unclear' in pred_str or 'partially true' in pred_str:
            return 0
        else:
            return 0

    def _prepare_feature_matrix(self, main_df):
        """
        Merge cached suspect features back into the main dataframe.

        Parameters
        ----------
        main_df : pd.DataFrame
            DataFrame containing Original Tweets.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature columns attached.
        """
        main = main_df.copy()

        for feature_name, feature_list in self.anomaly_detector.suspect_features.items():
            arr = np.asarray(feature_list)

            # If stored as [full_matrix], unwrap it
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

            if arr.shape[0] != len(main) and arr.shape[1] == len(main):
                arr = arr.T

            if arr.shape[0] != len(main):
                raise ValueError(
                    f"Feature '{feature_name}' has incompatible shape {arr.shape} "
                    f"for dataset length {len(main)}."
                )

            main[feature_name] = list(arr)

        return main

    def _build_training_data(self, merged):
        """
        Build feature/label tuples for each cached feature type.
        """
        training_data = {}
        labels = merged['label'].values

        for feature_name in self.anomaly_detector.suspect_features:
            features = np.vstack(merged[feature_name].values)
            training_data[feature_name] = (features, labels)

        return training_data

    def _estimate_feedback_rates(self, merged):
        """
        Estimate weak feedback rates from reviewed items.

        Since this pipeline does not yet store explicit false-positive review logs,
        we use fact-check labels as a proxy signal.
        """
        labels = merged['label'].values

        if len(labels) == 0:
            return 0.0, None

        verified_anomaly_rate = float(labels.mean())
        false_positive_rate = 1.0 - verified_anomaly_rate
        missed_anomaly_rate = None

        return false_positive_rate, missed_anomaly_rate

    def update_system(self):
        """
        Update the system using fact-check results and cached anomaly features.

        Returns
        -------
        dict
            Summary of model updates and threshold state.
        """
        self.load_thresholds()

        if not os.path.exists('fact_check_results.csv'):
            raise FileNotFoundError("fact_check_results.csv was not found.")

        fact_check_df = pd.read_csv('fact_check_results.csv')

        required_cols = {'Original Tweets', 'Fact_Check_Prediction'}
        missing_cols = required_cols - set(fact_check_df.columns)
        if missing_cols:
            raise ValueError(
                f"fact_check_results.csv is missing required columns: {missing_cols}"
            )

        fact_check_df = fact_check_df.copy()
        fact_check_df['label'] = fact_check_df['Fact_Check_Prediction'].apply(
            self._map_fact_check_label
        )
        fact_check_df = fact_check_df.dropna(subset=['label'])
        fact_check_df['label'] = fact_check_df['label'].astype(int)

        if 'Original Tweets' not in self.preprocessor.data.columns:
            raise ValueError("Preprocessor data does not contain 'Original Tweets'.")

        main = self.preprocessor.data[['Original Tweets']].copy()
        main = self._prepare_feature_matrix(main)

        merged = fact_check_df[['Original Tweets', 'label']].merge(
            main,
            on='Original Tweets',
            how='inner'
        )

        if merged.empty:
            raise ValueError("No verified tweets found in main data after merging.")

        training_data = self._build_training_data(merged)

        false_positive_rate, missed_anomaly_rate = self._estimate_feedback_rates(merged)

        self.update_thresholds(
            false_positive_rate=false_positive_rate,
            missed_anomaly_rate=missed_anomaly_rate
        )

        hybrid_results = self.update_hybrid_model(training_data)
        autoencoder_results = self.update_autoencoder(training_data)

        self.save_thresholds()

        return {
            'hybrid': hybrid_results,
            'autoenc': autoencoder_results,
            'thresholds': self.anomaly_detector.thresholds,
            'feedback_summary': {
                'false_positive_rate_proxy': false_positive_rate,
                'missed_anomaly_rate_proxy': missed_anomaly_rate,
                'reviewed_samples': len(merged)
            }
        }

    def update_hybrid_model(self, training_data, test_size=0.2, random_state=42):
        """
        Update the hybrid classifier using a holdout split.

        Returns
        -------
        dict
            Accuracy metrics for each classifier and feature type.
        """
        results = {}

        for feature_name, (features, labels) in training_data.items():
            if len(np.unique(labels)) < 2:
                results[feature_name] = {
                    'warning': 'Only one class present; skipping evaluation.'
                }
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )

            dt_model = self.anomaly_detector.dt.fit(X_train, y_train)
            svm_model = self.anomaly_detector.svm.fit(X_train, y_train)

            X_train_nb = np.clip(X_train, 0, None)
            X_test_nb = np.clip(X_test, 0, None)
            nb_model = self.anomaly_detector.nb.fit(X_train_nb, y_train)

            dt_pred = dt_model.predict(X_test)
            svm_pred = svm_model.predict(X_test)
            nb_pred = nb_model.predict(X_test_nb)

            results[feature_name] = {
                'dt_accuracy': float(accuracy_score(y_test, dt_pred)),
                'svm_accuracy': float(accuracy_score(y_test, svm_pred)),
                'nb_accuracy': float(accuracy_score(y_test, nb_pred)),
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test))
            }

        return results

    def update_autoencoder(self, training_data, autoencoder_encoding_dim=128):
        """
        Refresh the autoencoder using normal samples only.

        Returns
        -------
        dict
            Summary reconstruction statistics for each feature type.
        """
        results = {}

        for feature_name, (features, labels) in training_data.items():
            normal_data = features[labels == 0]

            if len(normal_data) == 0:
                results[feature_name] = {
                    'warning': 'No normal samples available for autoencoder update.'
                }
                continue

            scores = self.anomaly_detector.autoencoder_anomaly_detection(
                data=normal_data,
                encoding_dim=autoencoder_encoding_dim,
                force_retrain=False,
                return_scores=True
            )

            results[feature_name] = {
                'normal_samples_used': int(len(normal_data)),
                'mean_reconstruction_score': float(np.mean(scores)),
                'max_reconstruction_score': float(np.max(scores)),
                'threshold_percentile': float(
                    self.anomaly_detector.thresholds.get('autoencoder', 95.0)
                )
            }

        return results