# data/downloaders.py
import os
import logging
import pandas as pd
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def download_financial_data(output_path: str = config.FINANCIAL_RAW_DATA_FILE, force: bool = False) -> str:
    """
    Downloads the LendingClub raw dataset (2007-2014) via kagglehub if not present.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and not force:
        log.info("Financial raw dataset already exists at: %s", output_path)
        return output_path

    log.info("Downloading Financial Dataset (LendingClub 2007-2014) via kagglehub...")
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    file_path = "loan_data_2007_2014.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sreekargv/lending-club-data",
        file_path,
    )
    
    df.to_csv(output_path, index=False)
    log.info("Financial dataset saved to: %s (Shape: %s)", output_path, df.shape)
    return output_path


def download_clinical_data(
    features_path: str = config.CLINICAL_FEATURES_FILE,
    targets_path: str = config.CLINICAL_TARGETS_FILE,
    force: bool = False
) -> tuple:
    """
    Downloads the UCI Diabetes 130-US Hospitals (ID: 296) dataset via ucimlrepo if not present.
    """
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(targets_path), exist_ok=True)

    if os.path.exists(features_path) and os.path.exists(targets_path) and not force:
        log.info("Clinical dataset files already exist at: %s and %s", features_path, targets_path)
        return features_path, targets_path

    log.info("Downloading Clinical Dataset (UCI Diabetes 130 ID: 296) via ucimlrepo...")
    from ucimlrepo import fetch_ucirepo

    diabetes = fetch_ucirepo(id=296)
    df_features = diabetes.data.features
    df_targets = diabetes.data.targets

    df_features.to_csv(features_path, index=False)
    df_targets.to_csv(targets_path, index=False)
    
    log.info("Clinical dataset saved to: %s and %s", features_path, targets_path)
    return features_path, targets_path