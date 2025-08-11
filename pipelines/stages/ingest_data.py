import logging
import pandas as pd

from zenml import step
from pathlib import Path
from src.logger.handlers import get_logger
from pandas.errors import EmptyDataError, ParserError


class IngestData:
    """
    Handles ingestion of data from a CSV file.
    """

    def __init__(self, data_path: str, verbose: bool = False) -> None:
        """
        Initializes the IngestData class with the specified CSV file path.

        Args:
            data_path (str): The file path to the CSV data.
        """
        self.data_path = Path(data_path)
        self.logger = get_logger(verbose)

    def get_data(self) -> pd.DataFrame:
        """
        Reads and returns data from the CSV file.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            EmptyDataError: If the file is empty.
            ParserError: If the file cannot be parsed as a CSV.
            UnicodeDecodeError: If file encoding is invalid.
        """
        if not self.data_path.exists():
            self.logger.error(f"File not found at path: {self.data_path}")
            raise FileNotFoundError(f"No file found at path: {self.data_path}")

        try:
            df = pd.read_csv(self.data_path)
            
            if df.empty:
                self.logger.warning(f"The file at {self.data_path} is empty.")
                raise EmptyDataError(f"The file at {self.data_path} is empty.")
            
            self.logger.info(f"Successfully ingested data from: {self.data_path}")
            
            return df

        except (ParserError, UnicodeDecodeError) as e:
            self.logger.error(f"Error parsing CSV file at {self.data_path}: {e}")
            raise e


@step
def ingest_data(data_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    ZenML step for ingesting data from a CSV file.

    Args:
        data_path (str): The file path to the CSV data.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyDataError: If the file is empty.
        ParserError: If the file cannot be parsed as a CSV.
        UnicodeDecodeError: If file encoding is invalid.
        Exception: For any other unexpected ingestion failure.
    """
    logger = get_logger(verbose)
    
    try:
        return IngestData(data_path).get_data()
    
    except Exception as e:
        logger.exception("Data ingestion failed.")
        raise e
