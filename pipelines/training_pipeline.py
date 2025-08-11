from pathlib import Path
from zenml import pipeline

from pipelines.stages.clean_data import clean_data
from pipelines.stages.ingest_data import ingest_data
from pipelines.stages.model_train import train_model
from pipelines.stages.model_evaluate import evaluate_model

from src.logger.handlers import get_logger
from src.config.settings import ConfigurationManager


CONFIG_FILE_PATH = Path("configs/config.yaml")

@pipeline(enable_cache=False)
def train_pipeline(data_path: str, verbose: bool, pretrained_path: str = '') -> dict:
    """
    Train model pipeline that ingests, cleans, trains, and evaluates the model.

    Args:
        data_path: the path to the data
    Returns:
        Dictionary containing the r2_score and rmse_score
    """
    config = ConfigurationManager(config_file_path=CONFIG_FILE_PATH)

    data = ingest_data(data_path)
    
    X_train, X_test, y_train, y_test = clean_data(df=data, config=[config.get_general_config(), config.get_data_config()])

    model = train_model(X_train, y_train, pretrained_path)

    precision_score, recall_score, f1_score, roc_auc_score = evaluate_model(model, X_test, y_test)

    return {"precision_score": precision_score, 
            "recall_score": recall_score, 
            "f1_score": f1_score, 
            "roc_auc_score": roc_auc_score}