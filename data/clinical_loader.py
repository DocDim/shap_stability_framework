# data/clinical_loader.py
import os
import logging
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from data.downloaders import download_clinical_data
from data.clinical_preprocessing import run_clinical_preprocessing_pipeline

log = logging.getLogger(__name__)


def load_clinical_dataset(
    features_path: str = config.CLINICAL_FEATURES_FILE,
    targets_path: str = config.CLINICAL_TARGETS_FILE,
    ids_path: str = config.CLINICAL_IDS_FILE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Ensures UCI Diabetes 130 raw files exist (downloading if missing) and
    executes the 6-step clinical preprocessing pipeline.
    """
    if not all(os.path.exists(p) for p in (features_path, targets_path, ids_path)):
        # Re-fetch all three files together so their rows stay aligned
        download_clinical_data(features_path, targets_path, ids_path, force=True)

    log.info("Loading clinical data via preprocessing pipeline...")
    return run_clinical_preprocessing_pipeline(features_path, targets_path, ids_path)
