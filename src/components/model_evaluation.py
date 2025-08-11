import logging
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score,
                             roc_auc_score)

class Evaluation(ABC):
    """
    Abstract class for defining strategy for evaluating models.
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            dict: Dictionary containing scores
        """
        pass


class Precision(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Mean Squared Error.")
            
            mse = precision_score(y_true, y_pred)

            logging.info(f"Mean Squared Error: {mse}")
            return mse
        
        except Exception as e:
            logging.error(f"Error in calculating Mean Squared Error: {e}")
            raise e

class Recall(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Mean Squared Error.")
            
            mse = recall_score(y_true, y_pred)

            logging.info(f"Mean Squared Error: {mse}")
            return mse
        
        except Exception as e:
            logging.error(f"Error in calculating Mean Squared Error: {e}")
            raise e

class F1(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Mean Squared Error.")
            
            mse = f1_score(y_true, y_pred)

            logging.info(f"Mean Squared Error: {mse}")
            return mse
        
        except Exception as e:
            logging.error(f"Error in calculating Mean Squared Error: {e}")
            raise e
        
class RocAuc(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Mean Squared Error.")
            
            mse = roc_auc_score(y_true, y_pred)

            logging.info(f"Mean Squared Error: {mse}")
            return mse
        
        except Exception as e:
            logging.error(f"Error in calculating Mean Squared Error: {e}")
            raise e