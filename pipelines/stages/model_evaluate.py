"""

"""

import logging
import pandas as pd

from zenml import step
from typing import Tuple, Annotated
from src.components.model_training import Model
from src.components.model_evaluation import (Precision,
                                             Recall,
                                             F1,
                                             RocAuc)

@step
def evaluate_model(model: Model,
                   X_test: pd.DataFrame, 
                   y_test: pd.DataFrame
                   ) -> Tuple[Annotated[float, "Precision Score"],
                              Annotated[float, "Recall Score"],
                              Annotated[float, "F1 Score"],
                              Annotated[float, "ROC AUC Score"]]: 
    """
    Trains the model on the ingested data.

    Args:
        X_train: the training data
        y_train: the test data
    """
    try:
        y_pred = model.predict(X_test)

        precision = Precision().calculate_scores(y_test, y_pred)
        recall = Recall().calculate_scores(y_test, y_pred)
        f1 = F1().calculate_scores(y_test, y_pred)
        roc_auc = RocAuc().calculate_scores(y_test, y_pred)

        return precision, recall, f1, roc_auc
    
    except Exception as e:
        logging.error(f"Error in training model.")
        raise e