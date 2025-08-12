import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    pass


@dataclass
class ProcessingConfig:
    test_split: float
    random_state: int
    scaler_type: str = "standard"
    max_null_percentage: float = 0.95
    min_samples: int = 100


class DataValidator:
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: set) -> None:
        if df.empty:
            raise DataProcessingError("DataFrame is empty")

        logger.info(f"DataFrame: {df}")
           
        if len(df) < 100:
            raise DataProcessingError(f"Insufficient data: {len(df)} rows (minimum: 100)")
            
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise DataProcessingError(f"Missing required columns: {missing_cols}")
            
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_usage_mb > 1000:
            logger.warning(f"Large DataFrame: {memory_usage_mb:.1f}MB")

    @staticmethod
    def validate_config(config: ProcessingConfig) -> None:
        if not 0 < config.test_split < 1:
            raise ValueError("test_split must be between 0 and 1")
            
        if not isinstance(config.random_state, int):
            raise ValueError("random_state must be an integer")
            
        if config.scaler_type not in ["standard", "robust", "none"]:
            raise ValueError("scaler_type must be 'standard', 'robust', or 'none'")


class FeatureEngineer:
    @staticmethod
    def create_time_features(df: pd.DataFrame, scaled_time_col: str = 'scaled_time') -> pd.DataFrame:
        if scaled_time_col not in df.columns:
            return df
            
        df = df.copy()
        
        try:
            scaler_params = df[scaled_time_col].describe()
            std_time = scaler_params['std']
            mean_time = scaler_params['mean']
            
            original_time = (df[scaled_time_col] * std_time + mean_time)
            df['hour'] = (original_time // 3600) % 24
            df['day_of_week'] = ((original_time // 86400) % 7).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            logger.info("Created time-based features: hour, day_of_week, is_weekend")
            
        except Exception as e:
            logger.warning(f"Failed to create time features: {e}")
            
        return df

    @staticmethod
    def create_fraud_features(df: pd.DataFrame, target_col: str = 'Class', 
                            window_size: int = 10) -> pd.DataFrame:
        if target_col not in df.columns:
            return df
            
        df = df.copy()
        
        try:
            df = df.sort_values('scaled_time') if 'scaled_time' in df.columns else df
            
            df[f'fraud_prev_{window_size}'] = (
                df[target_col].rolling(window=window_size, min_periods=1)
                .sum().fillna(0).astype(int)
            )
            
            df['fraud_rate'] = (
                df[f'fraud_prev_{window_size}'] / window_size
            ).fillna(0)
            
            logger.info(f"Created fraud-based features with window size {window_size}")
            
        except Exception as e:
            logger.warning(f"Failed to create fraud features: {e}")
            
        return df


class DataPreprocessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.scaler = None
        self.imputer = None
        
    def _get_scaler(self):
        if self.config.scaler_type == "standard":
            return StandardScaler()
        elif self.config.scaler_type == "robust":
            return RobustScaler()
        else:
            return None
            
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        DataValidator.validate_dataframe(df, {'Amount', 'Time', 'Class'})
        
        df = df.copy()
        df = df.sort_values('Time')
        
        scaler = self._get_scaler()
        if scaler:
            scaled_features = scaler.fit_transform(df[['Amount', 'Time']])
            df[['scaled_amount', 'scaled_time']] = scaled_features
            self.scaler = scaler
        else:
            df['scaled_amount'] = df['Amount']
            df['scaled_time'] = df['Time']
            
        df = df.drop(['Amount', 'Time'], axis=1)
        
        df = FeatureEngineer.create_time_features(df)
        df = FeatureEngineer.create_fraud_features(df)
        
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_percentage > self.config.max_null_percentage:
            raise DataProcessingError(f"Too many null values: {null_percentage:.1%}")
            
        logger.info(f"Preprocessing completed: {len(df)} rows, {len(df.columns)} columns")
        return df


class DataSplitter:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.imputer = SimpleImputer(strategy='mean')
        
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        DataValidator.validate_dataframe(df, {'Class'})
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        if X.isnull().any().any():
            logger.info("Imputing missing values")
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X), 
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = X
            
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            raise DataProcessingError("Need at least 2 classes for splitting")
            
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, 
                test_size=self.config.test_split,
                random_state=self.config.random_state,
                stratify=y
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y,
                test_size=self.config.test_split,
                random_state=self.config.random_state
            )
            
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test


class DataProcessor:
    def __init__(self, config: ProcessingConfig):
        DataValidator.validate_config(config)
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.splitter = DataSplitter(config)
        
    def process_data(self, df: pd.DataFrame, 
                    save_processed: bool = False,
                    output_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        try:
            logger.info("Starting data preprocessing")
            processed_df = self.preprocessor.preprocess(df)
            
            if save_processed and output_path:
                processed_df.to_csv(output_path, index=False)
                logger.info(f"Processed data saved to: {output_path}")
                
            logger.info("Starting data splitting")
            return self.splitter.split(processed_df)
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}", exc_info=True)
            raise DataProcessingError(f"Data processing failed: {str(e)}") from e