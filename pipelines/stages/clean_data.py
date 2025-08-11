import logging
import pandas as pd

from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.logger.handlers import get_logger

from src.components.data_processing import DataCleaner, DataPreprocessStrategy, DataSplitStrategy

@step
def clean_data(df: pd.DataFrame, 
               config: list, 
               save_df: bool = False, 
               verbose: bool = False
               ) -> Tuple[Annotated[pd.DataFrame, "X_train"], 
                          Annotated[pd.DataFrame, "X_test"],
                          Annotated[pd.Series, "y_train"],
                          Annotated[pd.Series, "y_test"]]:
    """
    Cleans and preprocesses the input DataFrame and splits it into training and testing sets.

    Args:
        df (pd.DataFrame): Raw input DataFrame to clean and preprocess.
        config (list): Configuration list containing general and data configs.

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Training features.
            - X_test (pd.DataFrame): Testing features.
            - y_train (pd.Series): Training labels.
            - y_test (pd.Series): Testing labels.

    Raises:
        ValueError: If required columns are missing or configurations are invalid.
        Exception: For other unexpected errors during data cleaning and splitting.
    """
    logger = get_logger(verbose)

    try:
        logger.info("Starting data preprocessing...")
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaner = DataCleaner(df, preprocess_strategy, config)
        processed_data = data_cleaner.process_data()

        if save_df: processed_data.to_csv("data/processed/processed_data.csv", index=False)

        logger.info("Starting data splitting...")
        split_strategy = DataSplitStrategy()
        data_cleaner = DataCleaner(processed_data, split_strategy, config)
        X_train, X_test, y_train, y_test = data_cleaner.process_data()

        logger.info("Data cleaning and splitting completed successfully.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error during data cleaning step: {e}", exc_info=True)
        raise
