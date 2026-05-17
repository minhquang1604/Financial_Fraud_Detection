import os
import logging
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

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

RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "creditcard.csv"
)

REALTIME_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "realtime"
)

LABELED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labeled"
)


class LabelJoiner:

    def __init__(
        self,
        raw_data_path: str = RAW_DATA_PATH,
        realtime_dir: str = REALTIME_DIR,
        labeled_dir: str = LABELED_DIR
    ):
        self.raw_data_path = raw_data_path
        self.realtime_dir = realtime_dir
        self.labeled_dir = labeled_dir

        os.makedirs(self.realtime_dir, exist_ok=True)
        os.makedirs(self.labeled_dir, exist_ok=True)

        self._load_reference_data()

    # =========================
    # Load reference dataset
    # =========================
    def _load_reference_data(self):

        logger.info(f"Loading reference data from: {self.raw_data_path}")

        self.reference_df = pd.read_csv(self.raw_data_path)

        logger.info(f"Loaded {len(self.reference_df)} reference records")

        self.reference_df["join_key"] = (
            self.reference_df["Time"].astype(str)
            + "_"
            + self.reference_df["Amount"].astype(str)
        )

        self.reference_lookup = dict(
            zip(
                self.reference_df["join_key"],
                self.reference_df["Class"]
            )
        )

        logger.info("Reference lookup table created")

    # =========================
    # Create join key
    # =========================
    def _create_join_key(self, df: pd.DataFrame) -> pd.Series:

        return (
            df["Time"].astype(str)
            + "_"
            + df["Amount"].astype(str)
        )

    # =========================
    # Load realtime parquet files
    # =========================
    def _load_realtime_files(self) -> tuple[pd.DataFrame, List[str]]:

        files = [
            os.path.join(self.realtime_dir, f)
            for f in os.listdir(self.realtime_dir)
            if f.endswith(".parquet")
        ]

        if not files:
            raise FileNotFoundError("No realtime parquet files found")

        logger.info(f"Found {len(files)} realtime files")

        dfs = []

        for file in files:
            try:
                df = pd.read_parquet(file)

                logger.info(
                    f"Loaded {len(df)} records from {os.path.basename(file)}"
                )

                dfs.append(df)

            except Exception as e:
                logger.error(f"Failed to read {file}: {e}")

        if not dfs:
            raise ValueError("No valid parquet files could be loaded")

        combined_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"Combined dataframe contains {len(combined_df)} records")

        return combined_df, files

    # =========================
    # Join labels
    # =========================
    def join_labels(self, realtime_df: pd.DataFrame) -> pd.DataFrame:

        logger.info(f"Joining labels for {len(realtime_df)} records")

        df = realtime_df.copy()

        # Nếu đã có label sẵn
        if "Class" in df.columns and df["Class"].notna().all():
            logger.info("Realtime data already contains labels")
            return df

        df["join_key"] = self._create_join_key(df)

        df["Class"] = df["join_key"].map(self.reference_lookup)

        labeled_count = df["Class"].notna().sum()

        logger.info(f"Successfully joined {labeled_count}/{len(df)} labels")

        df = df.drop(columns=["join_key"])

        return df

    # =========================
    # Main batch processing
    # =========================
    def process_batch(self) -> Dict[str, Any]:

        try:
            realtime_df, processed_files = self._load_realtime_files()

            labeled_df = self.join_labels(realtime_df)

            # chỉ giữ sample có label
            labeled_df = labeled_df[labeled_df["Class"].notna()]

            if labeled_df.empty:
                return {
                    "success": False,
                    "error": "No labels could be joined"
                }

            version = datetime.now().strftime("%Y%m%d_%H%M%S")

            output_file = os.path.join(
                self.labeled_dir,
                f"labeled_batch_{version}.parquet"
            )

            labeled_df.to_parquet(output_file, index=False)

            logger.info(f"Labeled dataset saved to: {output_file}")

            return {
                "success": True,
                "output_file": output_file,
                "records": len(labeled_df),
                "files_processed": len(processed_files),
                "class_0": int((labeled_df["Class"] == 0).sum()),
                "class_1": int((labeled_df["Class"] == 1).sum())
            }

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

            return {
                "success": False,
                "error": str(e)
            }


# =========================
# Entry function
# =========================
def run_label_joiner() -> Dict[str, Any]:
    joiner = LabelJoiner()
    return joiner.process_batch()


# =========================
# CLI execution
# =========================
if __name__ == "__main__":

    result = run_label_joiner()

    print("\n" + "=" * 60)

    if result.get("success"):

        print("LABEL JOINER SUCCESS")
        print("=" * 60)

        print(f"Output File: {result['output_file']}")
        print(f"Records: {result['records']}")
        print(f"Files Processed: {result['files_processed']}")
        print(f"Class 0: {result['class_0']}")
        print(f"Class 1: {result['class_1']}")

    else:

        print("LABEL JOINER FAILED")
        print("=" * 60)
        print(f"Error: {result.get('error')}")

    print("=" * 60)