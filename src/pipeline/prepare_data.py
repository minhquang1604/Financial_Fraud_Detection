import os
import sys
import json
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from utils import engineer_features, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "train"
)

LABELED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labeled"
)

PROCESSED_LABELED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_labeled"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "mixed"
)


class DataMixer:

    DEFAULT_REF_RATIO = 0.75

    MIN_NEW_RECORDS = 100
    MIN_FRAUD_SAMPLES = 10

    def __init__(
        self,
        train_dir: str = TRAIN_DIR,
        labeled_dir: str = LABELED_DIR,
        processed_labeled_dir: str = PROCESSED_LABELED_DIR,
        output_dir: str = OUTPUT_DIR,
        ref_ratio: float = DEFAULT_REF_RATIO
    ):

        self.train_dir = train_dir
        self.labeled_dir = labeled_dir
        self.processed_labeled_dir = processed_labeled_dir
        self.output_dir = output_dir

        self.ref_ratio = ref_ratio
        self.new_ratio = 1.0 - ref_ratio

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_labeled_dir, exist_ok=True)

    def load_train_data(self) -> pd.DataFrame:

        logger.info(
            "Loading historical training data..."
        )

        files = [
            f for f in os.listdir(self.train_dir)
            if f.endswith(".parquet")
        ]

        if not files:

            raise FileNotFoundError(
                "No train parquet files found"
            )

        files = sorted(files)

        filepath = os.path.join(
            self.train_dir,
            files[-1]
        )

        df = pd.read_parquet(filepath)

        logger.info(
            f"Loaded train dataset: "
            f"{len(df)} records"
        )

        if "Class" not in df.columns:

            raise ValueError(
                "Training dataset missing 'Class' column"
            )

        df = df[df["Class"].notna()]

        return df

    def load_labeled_data(
        self
    ) -> Tuple[pd.DataFrame, List[str]]:

        logger.info(
            "Loading newly labeled data..."
        )

        files = [
            f for f in os.listdir(self.labeled_dir)
            if f.endswith(".parquet")
        ]

        if not files:

            logger.warning(
                "No labeled parquet files found"
            )

            return pd.DataFrame(), []

        full_paths = [
            os.path.join(self.labeled_dir, f)
            for f in files
        ]

        dfs = []

        for filepath in full_paths:

            try:

                df = pd.read_parquet(filepath)

                if "Class" not in df.columns:

                    logger.warning(
                        f"Skipping unlabeled file: "
                        f"{os.path.basename(filepath)}"
                    )

                    continue

                df = df[df["Class"].notna()]

                logger.info(
                    f"Loaded labeled batch: "
                    f"{len(df)} records from "
                    f"{os.path.basename(filepath)}"
                )

                dfs.append(df)

            except Exception as e:

                logger.error(
                    f"Failed loading "
                    f"{filepath}: {e}"
                )

        if not dfs:

            logger.warning(
                "No valid labeled data found"
            )

            return pd.DataFrame(), []

        combined_df = pd.concat(
            dfs,
            ignore_index=True
        )

        logger.info(
            f"Combined labeled dataset: "
            f"{len(combined_df)} records"
        )

        return combined_df, full_paths

    def validate_new_data(
        self,
        df: pd.DataFrame
    ):

        if len(df) < self.MIN_NEW_RECORDS:

            raise ValueError(
                f"Not enough new labeled data. "
                f"Required: {self.MIN_NEW_RECORDS}, "
                f"Found: {len(df)}"
            )

        fraud_count = (
            df["Class"] == 1
        ).sum()

        if fraud_count < self.MIN_FRAUD_SAMPLES:

            raise ValueError(
                f"Not enough fraud samples. "
                f"Required: {self.MIN_FRAUD_SAMPLES}, "
                f"Found: {fraud_count}"
            )

    def mix_data(
        self,
        train_df: pd.DataFrame,
        labeled_df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info("=" * 60)
        logger.info("DATA MIXING")
        logger.info(
            f"Historical ratio: "
            f"{self.ref_ratio:.0%}"
        )
        logger.info(
            f"New labeled ratio: "
            f"{self.new_ratio:.0%}"
        )
        logger.info("=" * 60)

        if labeled_df.empty:

            logger.warning(
                "No new labeled data found"
            )

            return train_df.copy()

        train_size = len(train_df)

        target_new_size = int(
            train_size
            * self.new_ratio
            / self.ref_ratio
        )

        if target_new_size > len(labeled_df):

            logger.warning(
                "Not enough labeled data to satisfy "
                "target ratio. Using all available "
                "labeled records."
            )

            target_new_size = len(labeled_df)

        sampled_new_df = labeled_df.sample(
            n=target_new_size,
            random_state=42
        )

        logger.info(
            f"Sampled {len(sampled_new_df)} "
            f"new labeled records"
        )

        mixed_df = pd.concat(
            [train_df, sampled_new_df],
            ignore_index=True
        )

        mixed_df = mixed_df.sample(
            frac=1.0,
            random_state=42
        ).reset_index(drop=True)

        logger.info(
            f"Final mixed dataset: "
            f"{len(mixed_df)} records"
        )

        logger.info(
            f"Class 0: "
            f"{(mixed_df['Class'] == 0).sum()}"
        )

        logger.info(
            f"Class 1: "
            f"{(mixed_df['Class'] == 1).sum()}"
        )

        return mixed_df

    def save_dataset(
        self,
        df: pd.DataFrame
    ) -> str:

        version = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_file = os.path.join(
            self.output_dir,
            f"mixed_train_{version}.parquet"
        )

        df.to_parquet(
            output_file,
            index=False
        )

        stats = {
            "version": version,
            "total_records": len(df),
            "class_0": int(
                (df["Class"] == 0).sum()
            ),
            "class_1": int(
                (df["Class"] == 1).sum()
            ),
            "historical_ratio": self.ref_ratio,
            "new_labeled_ratio": self.new_ratio,
            "created_at": datetime.now().isoformat()
        }

        stats_file = os.path.join(
            self.output_dir,
            f"stats_{version}.json"
        )

        with open(stats_file, "w") as f:

            json.dump(
                stats,
                f,
                indent=2
            )

        logger.info(
            f"Saved mixed dataset: "
            f"{output_file}"
        )

        return output_file

    def archive_processed_files(
        self,
        files: List[str]
    ):

        for filepath in files:

            try:

                destination = os.path.join(
                    self.processed_labeled_dir,
                    os.path.basename(filepath)
                )

                shutil.move(
                    filepath,
                    destination
                )

                logger.info(
                    f"Archived labeled file: "
                    f"{os.path.basename(filepath)}"
                )

            except Exception as e:

                logger.error(
                    f"Failed archiving file "
                    f"{filepath}: {e}"
                )

    def prepare_training_data(
        self
    ) -> Tuple[pd.DataFrame, str]:

        train_df = self.load_train_data()

        labeled_df, labeled_files = (
            self.load_labeled_data()
        )

        if not labeled_df.empty:

            self.validate_new_data(
                labeled_df
            )

        mixed_df = self.mix_data(
            train_df,
            labeled_df
        )

        logger.info(
            "Running feature engineering..."
        )

        mixed_df = engineer_features(
            mixed_df
        )

        feature_cols = get_feature_columns()

        missing_features = [
            col for col in feature_cols
            if col not in mixed_df.columns
        ]

        if missing_features:

            logger.warning(
                f"Missing features: "
                f"{missing_features}"
            )

        output_file = self.save_dataset(
            mixed_df
        )

        if labeled_files:

            self.archive_processed_files(
                labeled_files
            )

        return mixed_df, output_file


def run_prepare_data(
    ref_ratio: float = 0.75
) -> Dict[str, Any]:

    mixer = DataMixer(
        ref_ratio=ref_ratio
    )

    mixed_df, output_file = (
        mixer.prepare_training_data()
    )

    return {
        "success": True,
        "output_file": output_file,
        "historical_ratio": mixer.ref_ratio,
        "new_labeled_ratio": mixer.new_ratio,
        "total_records": len(mixed_df),
        "class_0": int(
            (mixed_df["Class"] == 0).sum()
        ),
        "class_1": int(
            (mixed_df["Class"] == 1).sum()
        )
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prepare retraining dataset"
    )

    parser.add_argument(
        "--ref-ratio",
        type=float,
        default=0.75,
        help="Historical data ratio"
    )

    args = parser.parse_args()

    result = run_prepare_data(
        ref_ratio=args.ref_ratio
    )

    print("\n" + "=" * 60)
    print("PREPARE DATA RESULT")
    print("=" * 60)

    print(
        f"Success: "
        f"{result['success']}"
    )

    print(
        f"Output File: "
        f"{result['output_file']}"
    )

    print(
        f"Historical Ratio: "
        f"{result['historical_ratio']:.0%}"
    )

    print(
        f"New Labeled Ratio: "
        f"{result['new_labeled_ratio']:.0%}"
    )

    print(
        f"Total Records: "
        f"{result['total_records']}"
    )

    print(
        f"Class 0: "
        f"{result['class_0']}"
    )

    print(
        f"Class 1: "
        f"{result['class_1']}"
    )

    print("=" * 60)