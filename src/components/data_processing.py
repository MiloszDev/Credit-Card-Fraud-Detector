"""
Module for preprocessing and splitting fraud detection data using strategy pattern.
"""

import logging
from typing import Union, Tuple, Dict
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataStrategy(ABC):
    """
    Abstract base class for data processing strategies.
    """

    @abstractmethod
    def process_data(
        self, data: pd.DataFrame, config: list
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Process the data according to a specific strategy.

        Args:
            data (pd.DataFrame): Input dataset.
            config (list): Configuration settings.

        Returns:
            Processed data, which can be a DataFrame or a tuple of train/test splits.
        """
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for preprocessing data by scaling features and creating additional engineered features.
    """

    def process_data(self, data: pd.DataFrame, config: list) -> pd.DataFrame:
        """
        Scale 'Amount' and 'Time' columns, drop originals, and add engineered features.

        Args:
            data (pd.DataFrame): Raw input data.
            config (list): Config list; not used here directly but kept for interface compatibility.

        Returns:
            pd.DataFrame: Preprocessed data with scaled features and engineered columns.

        Raises:
            ValueError: If expected columns are missing.
        """
        try:
            data = data.copy()

            required_columns = {'Amount', 'Time', 'Class'}
            if not required_columns.issubset(data.columns):
                raise ValueError(f"Missing required columns: {required_columns - set(data.columns)}")

            data = data.sort_values('Time')  # Ensure chronological order for rolling calculation

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(data[['Amount', 'Time']])
            data[['scaled_amount', 'scaled_time']] = scaled_features
            data = data.drop(['Amount', 'Time'], axis=1)

            std_time, mean_time = scaler.scale_[1], scaler.mean_[1]
            data['hour'] = ((data['scaled_time'] * std_time + mean_time) // 3600).astype(int)

            data['fraud_prev_10'] = data['Class'].rolling(window=10).sum().fillna(0)

            logger.info("Data preprocessing completed successfully.")
            return data

        except Exception as e:
            logger.error(f"Failed during preprocessing: {e}", exc_info=True)
            raise


class DataSplitStrategy(DataStrategy):
    """
    Strategy for splitting the dataset into training and testing sets.
    """

    def process_data(
        self, data: pd.DataFrame, config: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Impute missing values, split data into train and test sets.

        Args:
            data (pd.DataFrame): Preprocessed data.
            config (list): Configuration containing general and data configs.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train/test splits for features and labels.

        Raises:
            ValueError: If required columns are missing or config values are invalid.
        """
        try:
            data = data.copy()

            if 'Class' not in data.columns:
                raise ValueError("Missing target column 'Class'.")

            if not isinstance(config, list) or len(config) != 2:
                raise ValueError("Expected 'config' to be a list of two dictionaries: [general_config, data_config].")

            general_config, data_config = config

            if not isinstance(general_config, dict) or not isinstance(data_config, dict):
                raise TypeError("Config items must be dictionaries.")

            test_size = data_config.get('test_split')
            random_state = general_config.get('seed')

            if not isinstance(test_size, float) or not (0 < test_size < 1):
                raise ValueError("Config 'test_split' must be a float between 0 and 1.")

            if not isinstance(random_state, int):
                raise ValueError("Config 'seed' must be an integer.")

            X = data.drop('Class', axis=1)
            y = data['Class']

            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
            )

            logger.info(f"Data split into train/test with test size = {test_size}")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error during train-test split: {e}", exc_info=True)
            raise


class DataCleaner:
    """
    Applies a specified data processing strategy to the dataset.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy, config: list) -> None:
        """
        Initialize the DataCleaner with data, strategy, and config.

        Args:
            data (pd.DataFrame): Dataset to process.
            strategy (DataStrategy): Processing strategy instance.
            config (list): Configuration list.
        """
        self.data = data
        self.strategy = strategy
        self.config = config

    def process_data(
        self,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Execute the assigned data processing strategy.

        Returns:
            Processed data as defined by the strategy.
        """
        try:
            return self.strategy.process_data(self.data, self.config)
        except Exception as e:
            logger.error(f"Data processing failed in DataCleaner: {e}", exc_info=True)
            raise
