import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated, Union

from src.logger.handlers import get_logger
from src.components.data_processing import DataProcessor, ProcessingConfig


@step
def clean_data(
    df: pd.DataFrame, 
    config: list, 
    save_df: bool = False
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"], 
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    logger = get_logger(f"{__name__}.clean_data")
    
    try:
        if not isinstance(config, list) or len(config) != 2:
            raise ValueError("Config must be a list of [general_config, data_config]")
            
        general_config, data_config = config
        
        processing_config = ProcessingConfig(
            test_split=data_config["test_split"],
            random_state=general_config["seed"],
            scaler_type="standard",
            max_null_percentage=0.95,
            min_samples=100
        )
        
        processor = DataProcessor(processing_config)
        
        output_path = "data/processed/processed_data.csv" if save_df else None
        
        X_train, X_test, y_train, y_test = processor.process_data(
            df, 
            save_processed=save_df,
            output_path=output_path
        )
        
        logger.info("Data cleaning and splitting completed successfully")
        logger.info(f"Training set: {len(X_train)} samples, {len(X_train.columns)} features")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Data cleaning step failed: {e}", exc_info=True)
        raise