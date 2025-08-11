import logging
import pandas as pd

from zenml import step

from src.components.model_training import Model
from src.components.model_training import PPOAgent

@step
def train_model(X_train: pd.DataFrame, 
                y_train: pd.DataFrame,
                pretrained_path: str = None
                ) -> Model:
    """
    Trains the model on the ingested data.

    Args:
        X_train: the training data
        y_train: the test data
    """
    try:
        agent = PPOAgent()

        return agent.train_model(X_train, y_train) if not pretrained_path else agent.load_model(pretrained_path)
    
    except Exception as e:
        logging.error(f"Error in training model.")
        raise e