# data/clinical_loader.py
import os
import logging
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from data.downloaders import download_clinical_data
from data.clinical_preprocessing import run_clinical_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_clinical_dataset(
    features_path: str = config.CLINICAL_FEATURES_FILE,
    targets_path: str = config.CLINICAL_TARGETS_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Ensures UCI Diabetes 130 raw files exist (downloading if missing) and 
    executes the 6-step clinical preprocessing pipeline.
    """
    if not (os.path.exists(features_path) and os.path.exists(targets_path)):
        download_clinical_data(features_path=features_path, targets_path=targets_path)

    log.info("Loading clinical data via preprocessing pipeline...")
    return run_clinical_preprocessing_pipeline(features_path, targets_path)