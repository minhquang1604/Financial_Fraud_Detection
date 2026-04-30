import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from utils import engineer_features, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELED_DIR = os.path.join(PROJECT_ROOT, "data", "labeled")
LIVE_DIR = os.path.join(PROJECT_ROOT, "data", "live")
DVC_DIR = os.path.join(PROJECT_ROOT, "data", "dvc")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "mixed")


class DataMixer:
    DEFAULT_MIX_RATIO = 0.75

    def __init__(
        self,
        labeled_dir: str = LABELED_DIR,
        live_dir: str = LIVE_DIR,
        output_dir: str = OUTPUT_DIR,
        mix_ratio: float = DEFAULT_MIX_RATIO
    ):
        self.labeled_dir = labeled_dir
        self.live_dir = live_dir
        self.output_dir = output_dir
        self.mix_ratio = mix_ratio
        self.ref_ratio = 1.0 - mix_ratio

        os.makedirs(output_dir, exist_ok=True)

    def load_reference_data(self, version: Optional[str] = None) -> pd.DataFrame:
        logger.info("Loading reference data...")

        if not os.path.exists(self.labeled_dir):
            logger.warning(f"Labeled dir not found: {self.labeled_dir}")
            return pd.DataFrame()

        files = sorted([f for f in os.listdir(self.labeled_dir) if f.endswith('.parquet')])
        if not files:
            logger.warning("No labeled data files found")
            return pd.DataFrame()

        if version:
            filepath = os.path.join(self.labeled_dir, f"labeled_batch_{version}.parquet")
        else:
            filepath = os.path.join(self.labeled_dir, files[-1])

        df = pd.read_parquet(filepath)
        logger.info(f"Loaded reference: {len(df)} records from {os.path.basename(filepath)}")

        if "Class" not in df.columns:
            df = df[df["Class"].notna()]

        return df

    def load_live_data(self) -> pd.DataFrame:
        logger.info("Loading live data...")

        if not os.path.exists(self.live_dir):
            logger.warning(f"Live dir not found: {self.live_dir}")
            return pd.DataFrame()

        live_file = os.path.join(self.live_dir, "live_predictions.parquet")
        if not os.path.exists(live_file):
            logger.warning("No live predictions file found")
            return pd.DataFrame()

        df = pd.read_parquet(live_file)
        logger.info(f"Loaded live data: {len(df)} records")

        if df.empty:
            return pd.DataFrame()

        df = df[df["Class"].notna()]
        logger.info(f"After filtering labeled: {len(df)} records")

        return df

    def mix_data(
        self,
        reference_df: pd.DataFrame,
        live_df: pd.DataFrame,
        ref_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        ratio = ref_ratio if ref_ratio is not None else self.ref_ratio

        logger.info("=" * 60)
        logger.info("DATA MIXING (Secret Sauce)")
        logger.info(f"Reference ratio: {ratio:.0%}")
        logger.info(f"Live ratio: {1-ratio:.0%}")
        logger.info("=" * 60)

        if reference_df.empty:
            logger.warning("No reference data, using 100% live data")
            return live_df

        if live_df.empty:
            logger.warning("No live data, using 100% reference data")
            return reference_df

        ref_size = len(reference_df)
        live_size = len(live_df)

        if ref_size == 0 or live_size == 0:
            if ref_size > 0:
                return reference_df.copy()
            return live_df.copy()

        target_live_size = int(ref_size * (1 - ratio) / ratio)
        if target_live_size > live_size:
            target_live_size = live_size

        live_sample = live_df.sample(n=target_live_size, random_state=42)
        logger.info(f"Sampled {len(live_sample)} from {live_size} live records")

        mixed_df = pd.concat([reference_df, live_sample], ignore_index=True)
        mixed_df = mixed_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        logger.info(f"Mixed data: {len(mixed_df)} total records")
        logger.info(f"Class distribution:")
        logger.info(f"  Class 0 (Normal): {(mixed_df['Class'] == 0).sum()}")
        logger.info(f"  Class 1 (Fraud): {(mixed_df['Class'] == 1).sum()}")

        return mixed_df

    def save_mixed_data(
        self,
        df: pd.DataFrame,
        version: Optional[str] = None
    ) -> str:
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = os.path.join(self.output_dir, f"mixed_train_{version}.parquet")
        df.to_parquet(filepath, index=False)

        logger.info(f"Saved mixed data to {filepath}")

        stat_file = os.path.join(self.output_dir, f"stats_{version}.json")
        import json
        with open(stat_file, 'w') as f:
            json.dump({
                "version": version,
                "total_records": len(df),
                "class_0": int((df["Class"] == 0).sum()),
                "class_1": int((df["Class"] == 1).sum()),
                "ref_ratio": self.ref_ratio,
                "live_ratio": self.mix_ratio,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)

        return filepath

    def prepare_training_data(self, version: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        reference_df = self.load_reference_data(version)
        live_df = self.load_live_data()

        mixed_df = self.mix_data(reference_df, live_df)

        mixed_df = engineer_features(mixed_df)

        feature_cols = get_feature_columns()
        for col in feature_cols:
            if col not in mixed_df.columns:
                logger.warning(f"Missing feature column: {col}")

        filepath = self.save_mixed_data(mixed_df)

        return mixed_df, filepath


def run_prepare_data(
    version: Optional[str] = None,
    ref_ratio: Optional[float] = None
) -> Dict[str, Any]:
    mixer = DataMixer(mix_ratio=1-ref_ratio if ref_ratio else 0.75)

    mixed_df, filepath = mixer.prepare_training_data(version)

    return {
        "success": True,
        "version": version,
        "filepath": filepath,
        "ref_ratio": mixer.ref_ratio,
        "live_ratio": mixer.mix_ratio,
        "total_records": len(mixed_df),
        "class_0": int((mixed_df['Class'] == 0).sum()),
        "class_1": int((mixed_df['Class'] == 1).sum())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare mixed training data")
    parser.add_argument("--version", type=str, default=None, help="Reference data version")
    parser.add_argument("--ref-ratio", type=float, default=0.75, help="Reference data ratio (default: 0.75)")
    args = parser.parse_args()

    result = run_prepare_data(version=args.version, ref_ratio=args.ref_ratio)

    print("\n" + "=" * 60)
    print("PREPARE DATA RESULT")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Version: {result['version']}")
    print(f"Filepath: {result['filepath']}")
    print(f"Reference ratio: {result['ref_ratio']:.0%}")
    print(f"Live ratio: {result['live_ratio']:.0%}")
    print(f"Total records: {result['total_records']}")
    print(f"Class 0 (Normal): {result['class_0']}")
    print(f"Class 1 (Fraud): {result['class_1']}")
    print("=" * 60)