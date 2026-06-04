from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
import numpy as np
import keras
import pandas as pd

class AnomalyDetector:
    def __init__(self):
        """
        Initialize the AnomalyDetector object.

        This class contains methods for anomaly detection using k-means, DBSCAN, Autoencoder, DecisionTreeClassifier, SVC, and MultinomialNB.

        Attributes
        ----------
        None

        Methods
        -------
        kmeans_anomaly_detection(data, n_clusters=5)
            Uses k-means clustering to detect anomalies in the data.
        dbscan_anomaly_detection(data, eps=0.5, min_samples=5)
            Uses DBSCAN to detect anomalies in the data.
        hybrid_dtsvmnb_anomaly_detection(data, labels, threshold=0.5)
            Uses a hybrid approach combining DecisionTreeClassifier, SVC, and MultinomialNB to detect anomalies in the data.
        autoencoder_anomaly_detection(data, threshold=0.5)
            Uses an autoencoder to detect anomalies in the data.
        """
        self.autoencoder = None
        # Maintain model state across runs
        self.dt = DecisionTreeClassifier(random_state=42)
        self.svm = LinearSVC(random_state=42)
        self.nb = MultinomialNB()
        self.suspect_features = {}
        self.thresholds = {
            'kmeans': 97.5,
            'autoencoder': 97.5,
            'hybrid': 97.5
        }
        self.target_alert_rate = 0.03   # desired anomaly proportion
        self.min_percentile = 90.0
        self.max_percentile = 99.5

    def _score_to_flags(self, scores, model_name):
        """
        Converts anomaly scores to flags based on a percentile threshold.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            The anomaly scores to convert to flags.
        model_name : str
            The name of the model generating the anomaly scores.

        Returns
        -------
        flags : array-like of shape (n_samples,)
            Boolean indicators of anomalies.
        cutoff : float
            The threshold used to determine anomalies.
        """
        percentile = self.thresholds.get(model_name, 95.0)
        cutoff = np.percentile(scores, percentile)
        return (scores > cutoff).astype(int), cutoff
    
    def estimate_dbscan_eps(self, data, min_samples=5, quantile=0.98):
        """
        Estimates the optimal epsilon value for DBSCAN clustering.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input data to estimate the optimal epsilon value from.
        min_samples : int, optional (default=5)
            The minimum number of samples required to form a dense region.
        quantile : float, optional (default=0.98)
            The percentile of the k-th nearest neighbor distances to use as the optimal epsilon value.

        Returns
        -------
        float
            The estimated optimal epsilon value for DBSCAN clustering.
        """
        if hasattr(data, "toarray"):
            data_dense = data.toarray()
        else:
            data_dense = np.asarray(data)

        n_samples = len(data_dense)
        if n_samples == 0:
            raise ValueError("DBSCAN received an empty dataset.")

        effective_neighbors = max(1, min(min_samples, n_samples))

        nn = NearestNeighbors(n_neighbors=effective_neighbors)
        nn.fit(data_dense)
        distances, _ = nn.kneighbors(data_dense)

        kth_distances = np.sort(distances[:, -1])
        return float(np.quantile(kth_distances, quantile))

    def kmeans_anomaly_detection(self, data, n_clusters=5, return_scores=False):
        """
        Uses k-means clustering to detect anomalies in the data.
        Adapts cluster count for small live batches.
        """
        if hasattr(data, "toarray"):
            data_dense = data.toarray()
        else:
            data_dense = np.asarray(data)

        n_samples = len(data_dense)
        if n_samples == 0:
            raise ValueError("KMeans received an empty dataset.")
        if n_samples == 1:
            # With only one sample, return a neutral score/flag
            if return_scores:
                return np.zeros(1, dtype=float)
            return np.zeros(1, dtype=int)

        # Never request more clusters than samples
        effective_clusters = min(n_clusters, n_samples)

        kmeans = KMeans(n_clusters=effective_clusters, random_state=42).fit(data_dense)
        distances = np.linalg.norm(
            data_dense - kmeans.cluster_centers_[kmeans.labels_],
            axis=1
        )

        if return_scores:
            return distances
        else:
            anomalies, cutoff = self._score_to_flags(distances, 'kmeans')
            return anomalies
    
    def dbscan_anomaly_detection(self, data, eps=None, min_samples=10, return_scores=False):
        """
        Uses DBSCAN clustering to detect anomalies in the data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input data to detect anomalies from.
        eps : float, optional (default=None)
            The maximum distance between two samples in a cluster. If None, the optimal value is estimated.
        min_samples : int, optional (default=10)
            The minimum number of samples required to form a dense region.
        return_scores : boolean, optional (default=False)
            Whether to return anomaly scores or boolean indicators of anomalies.

        Returns
        -------
        If return_scores is False:
            anomalies : array-like of shape (n_samples,)
                A boolean array where True values indicate anomalies and False values indicate normal data points.
        If return_scores is True:
            scores : array-like of shape (n_samples,)
                An array of anomaly scores for the input data.
        """
        if hasattr(data, "toarray"):
            data_dense = data.toarray()
        else:
            data_dense = np.asarray(data, dtype=np.float64)

        n_samples = len(data_dense)
        if n_samples == 0:
            raise ValueError("DBSCAN received an empty dataset.")

        # Tiny batches cannot support large min_samples
        effective_min_samples = max(1, min(min_samples, n_samples))

        if eps is None:
            eps = self.estimate_dbscan_eps(
                data_dense,
                min_samples=effective_min_samples,
                quantile=0.98
            )

        dbscan = DBSCAN(
            eps=eps,
            min_samples=effective_min_samples
        ).fit(data_dense)

        if return_scores:
            if len(dbscan.components_) > 0:
                nn = NearestNeighbors(n_neighbors=1).fit(dbscan.components_)
                distances, _ = nn.kneighbors(data_dense)
                scores = distances.ravel() / max(eps, 1e-9)
            else:
                scores = np.full(shape=data_dense.shape[0], fill_value=np.inf)
            return scores
        else:
            return (dbscan.labels_ == -1).astype(int)
    
    def autoencoder_anomaly_detection(self, data, encoding_dim=128, threshold=0.5, force_retrain=False, return_scores=False):
        """
        Uses an autoencoder to detect anomalies in the data.

        The autoencoder is trained to reconstruct the input data. Anomalies are identified
        by calculating the mean squared error between the input data and the reconstructed
        data, and labeling data points with an error greater than a specified threshold 
        as anomalies.

        Parameters
        ----------
        data : array-like or sparse matrix of shape (n_samples, n_features)
            The input data to detect anomalies from.
        threshold : float, optional (default=0.5)
            The percentile threshold for determining anomalies based on reconstruction 
            error. Data points with errors above this threshold are considered anomalies.
        force_retrain : boolean, optional (default=False)
            Whether to force retraining of the autoencoder.
        return_scores : boolean, optional (default=False)
            Whether to return anomaly scores or boolean indicators of anomalies.

        Returns
        -------
        If return_scores is False:
            anomalies : array-like of shape (n_samples,)
                A boolean array where True values indicate anomalies and False values 
                indicate normal data points.
        If return_scores is True:
            scores : array-like of shape (n_samples,)
                An array of anomaly scores for the input data.
        """
        # Convert supported sparse/pandas inputs into a dense float32 NumPy array
        if sparse.issparse(data):
            data = data.toarray()
        elif isinstance(data, pd.DataFrame):
            # Handles both normal and pandas sparse DataFrames
            data = data.to_numpy()
        elif hasattr(data, "to_dense"):
            # Handles pandas sparse structures that expose to_dense()
            data = data.to_dense()
            if hasattr(data, "to_numpy"):
                data = data.to_numpy()

        data = np.asarray(data, dtype=np.float32)

        # Safety check
        if data.ndim != 2:
            raise ValueError(f"Autoencoder expected 2D input, got shape {data.shape}")

        input_dim = data.shape[1]

        old_autoencoder = self.autoencoder
        shape_mismatch = False

        if old_autoencoder is not None:
            expected = old_autoencoder.input_shape
            shape_mismatch = (expected[-1] != input_dim)

        if force_retrain or old_autoencoder is None or shape_mismatch:
            autoencoder = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                keras.layers.Dense(encoding_dim, activation='relu'),
                keras.layers.Dense(input_dim, activation='sigmoid')
            ])

            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(
                data,
                data,
                epochs=20,
                batch_size=32,
                verbose=1
            )

            self.autoencoder = autoencoder
        else:
            autoencoder = self.autoencoder

        recon = autoencoder.predict(data, verbose=0)
        mse = ((data - recon) ** 2).mean(axis=1)

        if return_scores:
            return mse
        else:
            anomalies, cutoff = self._score_to_flags(mse, 'autoencoder')
            return anomalies

    def hybrid_dtsvmnb_anomaly_detection(self, data, labels, return_scores=False):      
        """
        Uses a hybrid approach combining DecisionTreeClassifier, SVC, and MultinomialNB to detect anomalies in the data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input data to detect anomalies from.
        labels : array-like of shape (n_samples,)
            The corresponding labels for the input data.
        return_scores : boolean, optional (default=False)
            Whether to return anomaly scores or boolean indicators of anomalies.

        Returns
        -------
        If return_scores is False:
            anomalies : array-like of shape (n_samples,)
                A boolean array where True values indicate anomalies and False values indicate normal data points.
        If return_scores is True:
            scores : array-like of shape (n_samples,)
                An array of anomaly scores for the input data.
        """
        X = data
        y = labels

        if sparse.issparse(X):
            X_dense = X.toarray()
            X_nb = np.clip(X.toarray(), 0, None)
        else:
            X_dense = np.asarray(X)
            X_nb = np.clip(X_dense, 0, None)

        dt = DecisionTreeClassifier(random_state=42)
        svm = LinearSVC(random_state=42)
        nb = MultinomialNB()

        dt.fit(X_dense, y)
        svm.fit(X, y)
        nb.fit(X_nb, y)

        # Decision Tree score
        dt_proba = dt.predict_proba(X_dense)
        if dt_proba.ndim == 2 and dt_proba.shape[1] > 2:
            dt_score = 1.0 - np.max(dt_proba, axis=1)
        else:
            dt_score = dt_proba[:, 1] if dt_proba.shape[1] == 2 else dt_proba.ravel()

        # LinearSVC score
        svm_raw = svm.decision_function(X)
        if svm_raw.ndim == 2:
            svm_score = 1.0 - (
                np.max(svm_raw, axis=1) - np.min(svm_raw, axis=1)
            ) / (np.ptp(svm_raw, axis=1) + 1e-9)
        else:
            svm_score = (svm_raw - svm_raw.min()) / (svm_raw.max() - svm_raw.min() + 1e-9)

        # Naive Bayes score
        nb_proba = nb.predict_proba(X_nb)
        if nb_proba.ndim == 2 and nb_proba.shape[1] > 2:
            nb_score = 1.0 - np.max(nb_proba, axis=1)
        else:
            nb_score = nb_proba[:, 1] if nb_proba.shape[1] == 2 else nb_proba.ravel()

        anomaly_score = (dt_score + svm_score + nb_score) / 3.0

        if return_scores:
            return anomaly_score
        else:
            anomalies, cutoff = self._score_to_flags(anomaly_score, 'hybrid')
            return anomalies
    
    def cache_suspect_features(self, feature_name, features, suspect_idx):
        """
        Caches the suspect features of a given feature name for future reference.

        Parameters
        ----------
        feature_name : str
            The name of the feature to cache.
        features : array-like of shape (n_samples,)
            The values of the specified feature.
        suspect_idx : array-like of shape (n_samples,)
            A boolean array indicating which samples are suspected anomalies.

        Returns
        -------
        None
        """

        if feature_name in ('tfidf', 'dtm', 'bert'):
            if sparse.issparse(features): # If the features are sparse
                full_matrix = features.toarray() # Convert to dense array
            elif hasattr(features, 'values'): # If the features are a pandas DataFrame
                full_matrix = features.values # Convert to numpy array
            else: # If the features are a numpy array
                full_matrix = np.array(features) # Keep it as is
            
            self.suspect_features[feature_name] = [full_matrix]

            return
        elif feature_name not in self.suspect_features:
            self.suspect_features[feature_name] = []
        
        if isinstance(features, pd.DataFrame):
            selected = features.iloc[suspect_idx]
        else:
            selected = features[suspect_idx]

        self.suspect_features[feature_name].append(selected)