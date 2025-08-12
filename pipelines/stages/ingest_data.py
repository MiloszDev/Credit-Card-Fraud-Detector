import os
import pandas as pd
from pathlib import Path
from typing import Optional
from zenml import step

from src.logger.handlers import get_logger
from pandas.errors import EmptyDataError, ParserError


class DataIngestionError(Exception):
    pass


class CSVIngestor:
    MAX_FILE_SIZE_MB = 500
    ALLOWED_EXTENSIONS = {'.csv', '.CSV'}
    
    def __init__(self, data_path: str, encoding: str = 'utf-8', 
                 max_rows: Optional[int] = None) -> None:
        self.data_path = self._validate_and_resolve_path(data_path)
        self.encoding = encoding
        self.max_rows = max_rows
        self.logger = get_logger(f"{__name__}.CSVIngestor")

    def _validate_and_resolve_path(self, data_path: str) -> Path:
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError("data_path must be a non-empty string")
            
        path = Path(data_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
            
        if not path.is_file():
            raise DataIngestionError(f"Path is not a file: {path}")
            
        if path.suffix not in self.ALLOWED_EXTENSIONS:
            raise DataIngestionError(
                f"Invalid file type: {path.suffix}. "
                f"Allowed types: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
            
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise DataIngestionError(
                f"File too large: {file_size_mb:.1f}MB. "
                f"Maximum allowed: {self.MAX_FILE_SIZE_MB}MB"
            )
            
        return path

    def _detect_encoding(self) -> str:
        try:
            import chardet
            with open(self.data_path, 'rb') as file:
                raw_data = file.read(10240)
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding
                
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}. Using utf-8")
            return 'utf-8'

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise EmptyDataError(f"The file at {self.data_path} is empty")
            
        if len(df.columns) > 1000:
            self.logger.warning(f"DataFrame has {len(df.columns)} columns - this seems excessive")
            
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            self.logger.warning(f"Found duplicate column names: {duplicate_cols}")
            
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if null_percentage > 50:
            self.logger.warning(f"DataFrame has {null_percentage:.1f}% null values")

    def load_data(self, chunk_size: Optional[int] = None) -> pd.DataFrame:
        try:
            self.logger.info(f"Loading data from: {self.data_path}")
            
            read_params = {
                'encoding': self.encoding,
                'nrows': self.max_rows,
                'low_memory': False,
            }
            
            if chunk_size:
                read_params['chunksize'] = chunk_size
                
            try:
                if chunk_size:
                    chunks = []
                    for chunk in pd.read_csv(self.data_path, **read_params):
                        chunks.append(chunk)
                        if self.max_rows and sum(len(c) for c in chunks) >= self.max_rows:
                            break
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(self.data_path, **read_params)
                    
            except UnicodeDecodeError:
                self.logger.warning(f"Encoding {self.encoding} failed, attempting auto-detection")
                detected_encoding = self._detect_encoding()
                read_params['encoding'] = detected_encoding
                df = pd.read_csv(self.data_path, **read_params)
                
            self._validate_dataframe(df)
            
            self.logger.info(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
            
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            self.logger.info(f"DataFrame memory usage: {memory_usage_mb:.2f}MB")
            
            return df
            
        except (EmptyDataError, ParserError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise DataIngestionError(f"Data ingestion failed: {str(e)}") from e


def load_csv_data(
    data_path: str,
    encoding: str = 'utf-8',
    max_rows: Optional[int] = None,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    ingestor = CSVIngestor(data_path, encoding=encoding, max_rows=max_rows)
    return ingestor.load_data(chunk_size=chunk_size)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    logger = get_logger(f"{__name__}.ingest_data")
    
    try:
        logger.info(f"Starting data ingestion for: {data_path}")
        
        ingestor = CSVIngestor(data_path)
        df = ingestor.load_data()
        
        logger.info("Data ingestion completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}", exc_info=True)
        raise