from pathlib import Path
from typing import Optional, TypedDict
from zenml import pipeline

from pipelines.stages.clean_data import clean_data
from pipelines.stages.ingest_data import ingest_data
from pipelines.stages.model_train import train_model
from pipelines.stages.model_evaluate import evaluate_model

from src.logger.handlers import get_logger
from src.config.settings import ConfigurationManager


CONFIG_FILE_PATH = Path("configs/config.yaml")


class PipelineResults(TypedDict):
    """Type definition for pipeline results."""
    precision_score: float
    recall_score: float
    f1_score: float
    roc_auc_score: float


@pipeline(enable_cache=True)
def train_pipeline(
    data_path: str, 
    pretrained_path: str = "",
    config_path: str = ""
) -> PipelineResults:
    """
    Train model pipeline that ingests, cleans, trains, and evaluates the model.

    Args:
        data_path (str): Path to the training data file
        pretrained_path (Optional[str]): Path to pretrained model. Defaults to None.
        config_path (Optional[Path]): Path to config file. Defaults to None (uses default).
        
    Returns:
        PipelineResults: Dictionary containing evaluation metrics
        
    Raises:
        FileNotFoundError: If data_path, pretrained_path, or config file doesn't exist
        ValueError: If input parameters are invalid
        RuntimeError: If pipeline execution fails
    """
    if not isinstance(data_path, str) or not data_path.strip():
        raise ValueError("data_path must be a non-empty string")

    
    if pretrained_path is not None and not isinstance(pretrained_path, str):
        raise TypeError("pretrained_path must be a string or None")

    logger = get_logger()
    
    try:
        logger.info(f"Starting training pipeline with data: {data_path}")
        
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        if pretrained_path and not Path(pretrained_path).exists():
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

        config_file = config_path or CONFIG_FILE_PATH
        config = ConfigurationManager(config_file_path=config_file)
        logger.info(f"Loaded configuration from: {config_file}")

        logger.info("Step 1: Data ingestion")
        data = ingest_data(data_path)
        
        logger.info("Step 2: Data cleaning and preprocessing")
        X_train, X_test, y_train, y_test = clean_data(
            df=data, 
            config=[config.get_general_config(), config.get_data_config()],
        )

        logger.info("Step 3: Model training")
        model = train_model(X_train, y_train, pretrained_path)

        logger.info("Step 4: Model evaluation")
        precision_score, recall_score, f1_score, roc_auc_score = evaluate_model(
            model, X_test, y_test
        )

        results: PipelineResults = {
            "precision_score": precision_score, 
            "recall_score": recall_score, 
            "f1_score": f1_score, 
            "roc_auc_score": roc_auc_score
        }
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Results: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e