import sys
sys.path.insert(0, 'Modules')
import os
from pathlib import Path
from Data_Collector import DataCollector
from Data_Preprocessor import DataPreprocessor
from Anomaly_Detection import AnomalyDetector
from Fact_Checker import FactChecker
from Feedback import Feedback
from Metrics_Evaluation import MetricsEvaluator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


class PipelineManager:
    def __init__(self, config):
        # e.g. file paths, model settings, API keys
        self.data_source = config["data_source"]
        self.output_dir = Path(config.get("output_dir") or ".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collector = DataCollector(self.data_source)
        self.preprocessor = DataPreprocessor(self.collector)
        self.anomaly_detector = AnomalyDetector()
        self.fact_checker = FactChecker(output_dir=str(self.output_dir))

    def _out(self, filename: str) -> str:
        return str(self.output_dir / filename)

    def run(self):
        # 1. Collect
        self.collector.load_data()

        # 2. Preprocess
        self.preprocessor.preprocess_data()
        print("Rows after preprocess:", len(self.preprocessor.data))
        print(self.preprocessor.data[['Tweet ID', 'Original Tweets', 'Disaster']].head(10))
        embeddings = self.preprocessor.bert_tokenize()
        dtm = self.preprocessor.calculate_document_term_matrix()
        tfidf = self.preprocessor.calculate_term_frequency_inverse_document_frequency_matrix()

        # Access and display sentiment and emotion scores from the preprocessed data
        sentiment_labels = self.preprocessor.data['Sentiment_Label'].values
        sentiment_scores = self.preprocessor.data['Sentiment_Score'].values
        emotion_scores = self.preprocessor.data['Emotion_Labels'].values

        sentiment_path = self._out("sentiment_emotion_scores.csv")
        if not os.path.exists(sentiment_path):
            pd.DataFrame(
                {
                    'Sentiment_Score': sentiment_scores,
                    'Sentiment_Label': sentiment_labels,
                    'Emotion_Labels': emotion_scores
                }
            ).to_csv(sentiment_path, index=False)

        unique_disasters = self.preprocessor.data['Disaster'].nunique()
        can_run_hybrid = unique_disasters >= 2

        if not can_run_hybrid:
            print(
                f"Skipping hybrid DT-SVMNB: only one Disaster class present "
                f"({self.preprocessor.data['Disaster'].unique().tolist()})"
            )

        # 3. Anomaly Detection
        anomaly_results = {
            'kmeans_tfidf': self.anomaly_detector.kmeans_anomaly_detection(tfidf, return_scores=False),
            'kmeans_dtm': self.anomaly_detector.kmeans_anomaly_detection(dtm, return_scores=False),
            'kmeans_bert': self.anomaly_detector.kmeans_anomaly_detection(embeddings, return_scores=False),
            'dbscan_tfidf': self.anomaly_detector.dbscan_anomaly_detection(tfidf, return_scores=False),
            'dbscan_dtm': self.anomaly_detector.dbscan_anomaly_detection(dtm, return_scores=False),
            'dbscan_bert': self.anomaly_detector.dbscan_anomaly_detection(embeddings, return_scores=False),
            'autoencoder_tfidf': self.anomaly_detector.autoencoder_anomaly_detection(tfidf, return_scores=False),
            'autoencoder_dtm': self.anomaly_detector.autoencoder_anomaly_detection(dtm, return_scores=False),
            'autoencoder_bert': self.anomaly_detector.autoencoder_anomaly_detection(embeddings, return_scores=False)
        }

        if can_run_hybrid:
            anomaly_results['hybrid_dtsvmnb_tfidf'] = self.anomaly_detector.hybrid_dtsvmnb_anomaly_detection(
                tfidf, self.preprocessor.data['Disaster'], return_scores=False
            )
            anomaly_results['hybrid_dtsvmnb_dtm'] = self.anomaly_detector.hybrid_dtsvmnb_anomaly_detection(
                dtm, self.preprocessor.data['Disaster'], return_scores=False
            )

        anomaly_results_list = list(anomaly_results.values())

        # Union of anomalies
        suspect_idx = []

        feature_types = {
            'tfidf': tfidf,
            'dtm': dtm,
            'bert': embeddings
        }

        for arr in anomaly_results_list:
            arr_np = np.asarray(arr).ravel()

            if arr_np.dtype == bool:
                suspect_idx.extend(np.where(arr_np)[0])
            elif np.issubdtype(arr_np.dtype, np.integer):
                suspect_idx.extend(np.where(arr_np == 1)[0])
            elif isinstance(arr, pd.Series):
                suspect_idx.extend(arr[arr.astype(bool)].index.tolist())
            else:
                print(f"Warning: Unsupported anomaly result type/dtype: {type(arr)}, {arr_np.dtype}")

        suspect_idx = list(set(suspect_idx))
        valid_suspect_idx = [idx for idx in suspect_idx if idx < len(self.preprocessor.data)]
        suspects = self.preprocessor.data.iloc[valid_suspect_idx]

        for name, matrix in feature_types.items():
            self.anomaly_detector.cache_suspect_features(
                name,
                matrix,
                valid_suspect_idx
            )

        # Save anomaly flag results
        max_len = max(len(x) for x in anomaly_results_list)
        anomaly_results_df = pd.DataFrame({
            name: np.pad(np.asarray(x).astype(int), (0, max_len - len(x)))
            for name, x in anomaly_results.items()
        })

        anomaly_results_path = self._out("anomaly_results.csv")
        if not os.path.exists(anomaly_results_path):
            anomaly_results_df.to_csv(anomaly_results_path, index=False)

        # Retrieve score outputs from all functions and store them in a CSV file
        scores = {
            'kmeans_tfidf': self.anomaly_detector.kmeans_anomaly_detection(tfidf, return_scores=True),
            'kmeans_dtm': self.anomaly_detector.kmeans_anomaly_detection(dtm, return_scores=True),
            'kmeans_bert': self.anomaly_detector.kmeans_anomaly_detection(embeddings, return_scores=True),
            'dbscan_tfidf': self.anomaly_detector.dbscan_anomaly_detection(tfidf, return_scores=True),
            'dbscan_dtm': self.anomaly_detector.dbscan_anomaly_detection(dtm, return_scores=True),
            'dbscan_bert': self.anomaly_detector.dbscan_anomaly_detection(embeddings, return_scores=True),
            'autoencoder_tfidf': self.anomaly_detector.autoencoder_anomaly_detection(tfidf, return_scores=True),
            'autoencoder_dtm': self.anomaly_detector.autoencoder_anomaly_detection(dtm, return_scores=True),
            'autoencoder_bert': self.anomaly_detector.autoencoder_anomaly_detection(embeddings, return_scores=True)
        }

        if can_run_hybrid:
            scores['hybrid_dtsvmnb_tfidf'] = self.anomaly_detector.hybrid_dtsvmnb_anomaly_detection(
                tfidf, self.preprocessor.data['Disaster'], return_scores=False
            )
            scores['hybrid_dtsvmnb_dtm'] = self.anomaly_detector.hybrid_dtsvmnb_anomaly_detection(
                dtm, self.preprocessor.data['Disaster'], return_scores=False
            )

        # Check lengths first (debug step)
        for name, arr in scores.items():
            print(f"{name}: {len(arr)}")

        # Find the minimum common length
        min_len = min(len(arr) for arr in scores.values())

        # Truncate all arrays to min_len and flatten to 1D
        aligned_scores = {
            key: np.asarray(val).ravel()[:min_len]
            for key, val in scores.items()
        }

        scores_df = pd.DataFrame(aligned_scores)
        scores_path = self._out("anomaly_scores.csv")
        scores_df.to_csv(scores_path, index=False)

        print(f"Saved {len(scores_df)} rows of aligned anomaly scores.")

        print("Total preprocessed rows:", len(self.preprocessor.data))
        print("Suspect_idx count:", len(suspect_idx))
        print("Valid suspect_idx count:", len(valid_suspect_idx))
        print("Suspects shape:", suspects.shape)
        print("Suspects columns:", list(suspects.columns))
        print(suspects[['Tweet ID', 'Original Tweets']].head(10))

        # 4. Fact-check auto (robust)
        verification = None
        fc_path = self._out("fact_check_results.csv")
        if os.path.exists(fc_path):
            os.remove(fc_path)

        try:
            verification = self.fact_checker.fact_check_tweets(suspects, reset_output=True)
            print("Verification rows:", len(verification))
            print("Unique verification Tweet IDs:", verification['Tweet ID'].nunique() if 'Tweet ID' in verification.columns else 'N/A')
        except Exception as e:
            import traceback
            print("fact_check_tweets raised an exception:")
            traceback.print_exc()
            raise

        if verification is None:
            if os.path.isfile(fc_path):
                try:
                    verification = pd.read_csv(fc_path)
                    print(f"Loaded verification DataFrame from {fc_path} (rows={len(verification)})")
                except Exception as e:
                    print("Failed to read fact_check_results.csv:", e)
                    verification = pd.DataFrame()
            else:
                print("No fact_check_results.csv found and fact_check_tweets returned None. verification set to empty DataFrame.")
                verification = pd.DataFrame()

        # 5. Feedback
        feedback = Feedback(
            self.fact_checker,
            self.anomaly_detector,
            self.collector.data_source,
            self.preprocessor
        )

        # 6. Metrics Evaluation / alignment
        fact_check_results = verification.copy()

        print("Predictions DataFrame columns (raw):")
        print(list(fact_check_results.columns))
        print("Predictions columns (repr):")
        print([repr(c) for c in fact_check_results.columns])
        print("\nShow head of predictions:")
        print(fact_check_results.head(5))

        def normalize_cols(df):
            df = df.copy()
            df.columns = (
                df.columns.astype(str)
                .str.replace('\ufeff', '', regex=False)
                .str.replace('\r', '', regex=False)
                .str.replace('\n', '', regex=False)
                .str.strip()
            )
            return df

        fact_check_results = normalize_cols(fact_check_results)
        self.preprocessor.data = normalize_cols(self.preprocessor.data)

        def find_col(df, candidates):
            cols = list(df.columns)
            low = [c.lower() for c in cols]
            for cand in candidates:
                cand_l = cand.lower()
                for i, c in enumerate(low):
                    if cand_l == c or cand_l in c or c in cand_l:
                        return cols[i]
            return None

        pred_candidates = ['fact_check_prediction', 'fact_check_pred', 'fact_check', 'factcheck', 'prediction', 'pred', 'label']
        text_candidates = ['original tweets', 'original tweet', 'tweet', 'text', 'tweet_text']
        score_candidates = ['score', 'prob', 'probability', 'confidence']

        pred_col = find_col(fact_check_results, pred_candidates)
        text_col = find_col(fact_check_results, text_candidates)
        score_col = find_col(fact_check_results, score_candidates)

        print("\nDetected columns in predictions:")
        print("pred_col:", pred_col)
        print("text_col (predictions):", text_col)
        print("score_col:", score_col)

        left_text_col = 'Original Tweets'
        if left_text_col not in self.preprocessor.data.columns:
            left_text_col = find_col(self.preprocessor.data, text_candidates)
            print("Fallback left_text_col:", left_text_col)

        if pred_col is None:
            raise KeyError(
                f"Could not auto-detect a predictions column in fact_check_results. "
                f"Columns: {list(fact_check_results.columns)}"
            )

        merged = None

        if text_col is not None and left_text_col is not None:
            fact_check_results[text_col] = fact_check_results[text_col].astype(str).str.strip()
            self.preprocessor.data[left_text_col] = self.preprocessor.data[left_text_col].astype(str).str.strip()

            cols_to_take = [text_col, pred_col] + (
                [score_col] if (score_col is not None and score_col in fact_check_results.columns) else []
            )

            preds_subset = fact_check_results[cols_to_take].copy()
            merged = self.preprocessor.data.merge(
                preds_subset,
                left_on=left_text_col,
                right_on=text_col,
                how='inner'
            )

            print("Merged on text rows:", len(merged))

        if merged is None or len(merged) == 0:
            print("Text merge failed or returned 0 rows. Trying safe merge on 'Tweet ID'...")

            def normalize_id_col(df, col_name):
                s = df[col_name].astype(str).str.strip()
                s = s.str.replace(r'\.0+$', '', regex=True)
                s = s.str.replace(',', '', regex=False).str.replace('"', '', regex=False)
                return s

            preds = fact_check_results.copy()
            data = self.preprocessor.data.copy()

            preds.columns = preds.columns.astype(str).str.strip()
            data.columns = data.columns.astype(str).str.strip()

            pred_id_col = 'Tweet ID'
            data_id_col = 'Tweet ID'

            preds['_tid_'] = normalize_id_col(preds, pred_id_col)
            data['_tid_'] = normalize_id_col(data, data_id_col)

            cols_needed = ['_tid_', pred_col] + (
                [score_col] if (score_col and score_col in preds.columns) else []
            )

            if pred_col not in preds.columns:
                raise KeyError(
                    f"Expected prediction column '{pred_col}' not found in predictions. "
                    f"Available: {list(preds.columns)}"
                )

            preds_subset = preds[cols_needed].copy().drop_duplicates(subset=['_tid_'], keep='first')
            merged = data.merge(preds_subset, on='_tid_', how='inner', suffixes=('_data', '_pred'))

            print("Merged_on_id rows:", len(merged))

        if merged is None or len(merged) == 0:
            raise ValueError("Could not align predictions with dataset using either text or Tweet ID.")

        merged_path = self._out("merged_data_results.csv")
        merged.to_csv(merged_path, index=False)
        print(f"Saved merged results to {merged_path}")

        return {
            "anomaly_results": anomaly_results,
            "output_files": {
                "anomaly_scores": self._out("anomaly_scores.csv"),
                "anomaly_results": self._out("anomaly_results.csv"),
                "fact_check_results": self._out("fact_check_results.csv"),
                "merged_data_results": self._out("merged_data_results.csv"),
                "sentiment_emotion_scores": self._out("sentiment_emotion_scores.csv"),
            }
        }