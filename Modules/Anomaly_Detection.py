from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
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
        self.svm = SVC(probability=True, kernel='linear', random_state=42)
        self.nb = MultinomialNB()
        self.suspect_features = {}

    def kmeans_anomaly_detection(self, data, n_clusters=5):
        """
        Uses k-means clustering to detect anomalies in the data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input data to detect anomalies from.
        n_clusters : int, optional (default=5)
            The number of clusters to form and the number of centroids to generate.

        Returns
        -------
        anomalies : array-like of shape (n_samples, )
            A boolean array where True values indicate anomalies and False values indicate normal data points.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)

        distances = np.linalg.norm(data - kmeans.cluster_centers_[kmeans.labels_], axis=1)
        threshold = np.percentile(distances, 95)
        anomalies = distances > threshold

        return anomalies
    
    def dbscan_anomaly_detection(self, data, eps=0.5, min_samples=5):
        """
        Uses DBSCAN clustering to detect anomalies in the data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input data to detect anomalies from.
        eps : float, optional (default=0.5)
            The maximum distance between two samples for them to be considered 
            as in the same neighborhood.
        min_samples : int, optional (default=5)
            The number of samples in a neighborhood for a point to be considered 
            as a core point. This includes the point itself.

        Returns
        -------
        anomalies : array-like of shape (n_samples,)
            A boolean array where True values indicate anomalies (noise points) 
            and False values indicate normal data points.
        """

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

        anomalies = dbscan.labels_ == -1

        return anomalies
    
    def autoencoder_anomaly_detection(self, data, encoding_dim = 128, threshold=0.5, force_retrain=False):
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

        Returns
        -------
        anomalies : array-like of shape (n_samples,)
            A boolean array where True values indicate anomalies and False values 
            indicate normal data points.
        """
        # Convert sparse matrix to dense array
        if isinstance(data, sparse.csr.csr_matrix):
            data = data.toarray()

        # Determine input dimension from data
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
            
        recon = autoencoder.predict(data)
        mse = ((data - recon) ** 2).mean(axis=1)
        threshold = np.percentile(mse, 95)
        anomalies = mse > threshold

        return anomalies

    def hybrid_dtsvmnb_anomaly_detection(self, data, labels, threshold=0.5):
        """
        Uses a hybrid approach combining a DecisionTreeClassifier, a Support Vector Machine (SVM), and a Multinomial Naive Bayes (NB) classifier to detect anomalies in the data.

        The anomaly score for each data point is calculated by summing the mean probabilities predicted by each classifier. Data points with an anomaly score above a specified percentile 
        threshold are labeled as anomalies.

        Parameters
        ----------
        data : array-like or sparse matrix of shape (n_samples, n_features)
            The input data to detect anomalies from.
        labels : array-like of shape (n_samples,)
            The labels for the input data. Used for training the classifiers.
        threshold : float, optional (default=0.5)
            The percentile threshold for determining anomalies based on the anomaly score. Data points with an anomaly score above this threshold are considered anomalies.

        Returns
        -------
        anomalies : array-like of shape (n_samples,)
            A boolean array where True values indicate anomalies and False values indicate normal data points.
        """
        # Initialize classifiers
        dt = DecisionTreeClassifier(random_state=42)
        svm = SVC(probability=True, kernel='linear', random_state=42)
        nb = MultinomialNB()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Fit models
        dt.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        nb.fit(X_train, y_train)

        # Predict probabilities
        dt_pred = dt.predict_proba(X_test)
        svm_pred = svm.predict_proba(X_test)
        nb_pred = nb.predict_proba(X_test)

        # Calculate anomaly score
        anomaly_score = np.mean(dt_pred, axis=1) + np.mean(svm_pred, axis=1) + np.mean(nb_pred, axis=1)

        # Set threshold
        threshold = np.percentile(anomaly_score, 95)

        # Detect anomalies
        anomalies = anomaly_score > threshold

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