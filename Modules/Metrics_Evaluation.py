import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings

class MetricsEvaluator:
    def __init__(self, verification_results, fact_check_results, dataset, label_column=None, auto_align=True):
        """
        verification_results: scores or probabilities or predictions (depending on your use)
        fact_check_results: model predicted labels (or predicted binary labels)
        dataset: ground-truth labels OR a dataframe containing labels
        label_column: if `dataset` is a DataFrame, name of the column with true labels
        auto_align: if True and both inputs are pandas objects with indices, align on index (inner join)
        """
        self.verification_results = verification_results
        self.fact_check_results = fact_check_results
        self.dataset = dataset
        self.label_column = label_column
        self.auto_align = auto_align

    def _to_1d_array(self, x, name="value"):
        """Convert pandas Series/DataFrame or list/np.ndarray to 1D numpy array. Raise helpful errors."""
        # Series
        if isinstance(x, pd.Series):
            return x.values, x.index
        # DataFrame with single column
        if isinstance(x, pd.DataFrame):
            if x.shape[1] == 1:
                return x.iloc[:, 0].values, x.index
            else:
                raise ValueError(f"{name} is a DataFrame with multiple columns; pass a Series or provide label_column.")
        # numpy or list-like
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                return x, None
            if x.ndim == 2 and x.shape[1] == 1:
                return x.ravel(), None
            raise ValueError(f"{name} numpy array has unexpected shape {x.shape}.")
        # list or other sequence
        if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
            return np.asarray(x), None

        raise TypeError(f"Unsupported type for {name}: {type(x)}")

    def _get_true_and_pred(self):
        # Extract y_true from dataset
        """
        Return y_true, y_pred, and y_score arrays from the attributes of `self`.

        If `self.dataset` is a DataFrame, it is expected to have a single column
        with ground-truth labels. If `self.label_column` is not None, it is used
        to select that column. Otherwise, the first column is used.

        If `self.auto_align` is True, attempt to align y_true and y_pred by
        index if they are pandas objects with indices. If y_score has an index
        and is present, align it as well.

        If lengths of y_true and y_pred do not match after alignment, raise
        ValueError.

        Returns (y_true, y_pred, y_score) where y_score is None if no scores
        were provided or alignment failed.
        """
        y_true_raw = self.dataset
        if isinstance(self.dataset, pd.DataFrame):
            if self.label_column is None:
                raise ValueError("dataset is a DataFrame. Pass label_column=<column name> that contains ground-truth labels.")
            y_true_raw = self.dataset[self.label_column]

        y_true, idx_true = self._to_1d_array(y_true_raw, name="y_true")
        y_pred, idx_pred = self._to_1d_array(self.fact_check_results, name="y_pred")
        y_score, idx_score = None, None
        # If verification_results likely contains scores/probs, convert too (used for roc_auc)
        try:
            y_score, idx_score = self._to_1d_array(self.verification_results, name="y_score")
        except Exception:
            # verification_results may be None or not needed — that's fine
            y_score = None

        # If both are pandas with indices and auto_align True, align on index
        if self.auto_align and idx_true is not None and idx_pred is not None:
            # idx_true and idx_pred are Index objects
            if not idx_true.equals(idx_pred):
                # attempt inner join alignment
                ser_true = pd.Series(y_true, index=idx_true)
                ser_pred = pd.Series(y_pred, index=idx_pred)
                joined = ser_true.align(ser_pred, join="inner")
                ser_true_aligned, ser_pred_aligned = joined
                if len(ser_true_aligned) == 0:
                    raise ValueError("After aligning on index, no overlapping rows found between y_true and y_pred.")
                warnings.warn(f"Aligned y_true and y_pred by index: lengths {len(y_true)} -> {len(ser_true_aligned)}")
                y_true = ser_true_aligned.values
                y_pred = ser_pred_aligned.values
                # align y_score if index available and present
                if y_score is not None and idx_score is not None:
                    ser_score = pd.Series(y_score, index=idx_score)
                    _, ser_score_aligned = ser_true_aligned.align(ser_score, join="inner")
                    if len(ser_score_aligned) != len(y_true):
                        warnings.warn("y_score could not be fully aligned with y_true. y_score will be ignored for ROC-AUC.")
                        y_score = None
                    else:
                        y_score = ser_score_aligned.values
        # final length check
        if len(y_true) != len(y_pred):
            raise ValueError(f"Found input variables with inconsistent numbers of samples: [{len(y_pred)}, {len(y_true)}]. "
                             "Make sure predictions and ground-truth labels match in length and order.")
        return y_true, y_pred, y_score

    def calculate_accuracy(self):
        """
        Computes the accuracy score, given true labels and predicted labels.

        Returns
        -------
        score : float
            The accuracy score of the positive class in binary classification or weighted average of the accuracy score of each class for the multiclass task.
        """
        
        y_true, y_pred, _ = self._get_true_and_pred()

        return accuracy_score(y_true, y_pred)

    def calculate_precision(self, average='binary'):      
        """
        Computes the precision score, also known as positive predictive value, given true labels and predicted labels.

        Parameters
        ----------
        average : str, optional (default='binary')
            This parameter is required for multiclass problems. If None, the scores for each class are returned. Else,
            this determines the type of averaging performed on the data.

        Returns
        -------
        score : float
            The precision score of the positive class in binary classification or weighted average of the precision score
            of each class for the multiclass task.
        """
        y_true, y_pred, _ = self._get_true_and_pred()
        # If not binary, user may prefer 'macro' or 'weighted'

        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_recall(self, average='binary'):
        """
        Computes the recall score, also known as sensitivity or true positive rate, given true labels and predicted labels.

        Parameters
        ----------
        average : str, optional (default='binary')
            This parameter is required for multiclass problems. If None, the scores for each class are returned. Else,
            this determines the type of averaging performed on the data.

        Returns
        -------
        score : float
            Recall of the positive class in binary classification or weighted average of the recall of each class for
            the multiclass task.
        """
        y_true, y_pred, _ = self._get_true_and_pred()

        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_f1_score(self, average='binary'):
        """
        Computes the F1 score, also known as balanced F-score or F-measure, given true labels and predicted labels.

        The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its
        best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are
        equal.

        Parameters
        ----------
        average : str, optional (default='binary')
            This parameter is required for multiclass problems. If None, the scores for each class are returned. Else,
            this determines the type of averaging performed on the data.

        Returns
        -------
        score : float
            F1 score of the positive class in binary classification or weighted average of the F1 score of each class for
            the multiclass task.
        """
        y_true, y_pred, _ = self._get_true_and_pred()
        
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def calculate_roc_auc_score(self):
        # roc_auc_score expects (y_true, y_score) where y_score are probabilities/scores
        """
        Computes the area under the receiver operating characteristic curve (ROC-AUC score)
        given true labels and the model's predicted probabilities/scores.

        Parameters
        ----------
        None

        Returns
        -------
        score : float
            The ROC-AUC score.

        Raises
        ------
        ValueError
            If no model scores/probabilities are provided in `verification_results`.
            If multiclass and the computation fails.
        """
        y_true, _, y_score = self._get_true_and_pred()
        if y_score is None:
            raise ValueError("roc_auc_score requires model scores/probabilities as the second argument. "
                             "Provide these in `verification_results` (not labels).")
        # If multiclass, sklearn needs shape (n_samples, n_classes) and multi_class param; keep simple here:
        if y_score.ndim > 1 and y_score.shape[1] > 1:
            # multiclass probability matrix
            try:
                return roc_auc_score(y_true, y_score, multi_class="ovr")
            except Exception as e:
                raise ValueError("Failed to compute multiclass ROC-AUC: " + str(e))
        else:
            return roc_auc_score(y_true, y_score)

    def calculate_confusion_matrix(self):
        """
        Compute confusion matrix to evaluate the accuracy of a classification.

        Parameters
        ----------
        None

        Returns
        -------
        confusion_matrix : array, shape = [n_classes, n_classes]
            Confusion matrix whose i-th row and j-th column entry indicates the number of samples
            from the i-th class labeled as j-th class.

        Notes
        -----
        In binary classification, the count of true negatives is `C[0,0]`, false positives is `C[0,1]`,
        false negatives is `C[1,0]` and true positives is `C[1,1]`. The confusion_matrix function will
        be removed in version 1.3. Please use sklearn.metrics.confusion_matrix instead.
        """
        y_true, y_pred, _ = self._get_true_and_pred()

        return confusion_matrix(y_true, y_pred)