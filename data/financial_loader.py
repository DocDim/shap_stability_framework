# data/financial_loader.py
import os
import logging
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from data.downloaders import download_financial_data
from data.financial_preprocessing import run_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_preprocessed_finance_dataset(
    filepath: str = config.FINANCIAL_RAW_DATA_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    if not os.path.exists(filepath):
        download_financial_data(output_path=filepath)

    log.info("Loading financial data via preprocessing pipeline from: %s", filepath)
    return run_preprocessing_pipeline(filepath)