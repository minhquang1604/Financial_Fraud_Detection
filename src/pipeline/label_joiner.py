import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from utils import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "creditcard.csv")
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
LABELED_DIR = os.path.join(PROJECT_ROOT, "data", "labeled")


class LabelJoiner:
    def __init__(
        self,
        raw_data_path: str = RAW_DATA_PATH,
        staging_dir: str = STAGING_DIR,
        labeled_dir: str = LABELED_DIR
    ):
        self.raw_data_path = raw_data_path
        self.staging_dir = staging_dir
        self.labeled_dir = labeled_dir
        
        os.makedirs(staging_dir, exist_ok=True)
        os.makedirs(labeled_dir, exist_ok=True)
        
        self._load_reference_data()
    
    def _load_reference_data(self):
        logger.info(f"Loading reference data from {self.raw_data_path}")
        self.reference_df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded {len(self.reference_df)} reference records")
    
    def _get_join_key(self, row: pd.Series) -> str:
        return f"{row['Time']}_{row['Amount']}"
    
    def join_labels(self, staging_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Joining labels for {len(staging_df)} staging records...")
        
        staging_df = staging_df.copy()
        
        if 'Class' in staging_df.columns and staging_df['Class'].notna().all():
            logger.info("Staging data already has labels")
            return staging_df
        
        reference_lookup = {}
        for _, row in self.reference_df.iterrows():
            key = self._get_join_key(row)
            reference_lookup[key] = row['Class']
        
        def lookup_class(r):
            key = self._get_join_key(r)
            return reference_lookup.get(key, np.nan)
        
        staging_df['Class'] = staging_df.apply(lookup_class, axis=1)
        
        labeled_count = staging_df['Class'].notna().sum()
        logger.info(f"Joined {labeled_count}/{len(staging_df)} labels")
        
        return staging_df
    
    def process_batch(self, staging_file: Optional[str] = None) -> Dict[str, Any]:
        try:
            if staging_file is None:
                files = [f for f in os.listdir(self.staging_dir) 
                        if f.endswith('.parquet') and 'staging' in f]
                if not files:
                    return {"success": False, "error": "No staging files found"}
                staging_file = os.path.join(self.staging_dir, files[0])
            
            if not os.path.exists(staging_file):
                return {"success": False, "error": f"File not found: {staging_file}"}
            
            staging_df = pd.read_parquet(staging_file)
            
            labeled_df = self.join_labels(staging_df)
            
            labeled_df = labeled_df[labeled_df['Class'].notna()]
            
            if labeled_df.empty:
                return {"success": False, "error": "No labels joined"}
            
            filename = os.path.basename(staging_file)
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.labeled_dir, 
                f"labeled_batch_{version}.parquet"
            )
            
            labeled_df.to_parquet(output_file, index=False)
            
            return {
                "success": True,
                "staging_file": staging_file,
                "output_file": output_file,
                "records": len(labeled_df),
                "class_0": int((labeled_df['Class'] == 0).sum()),
                "class_1": int((labeled_df['Class'] == 1).sum())
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return {"success": False, "error": str(e)}


def run_label_joiner(staging_file: Optional[str] = None) -> Dict[str, Any]:
    joiner = LabelJoiner()
    return joiner.process_batch(staging_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Label Joiner")
    parser.add_argument(
        "--staging-file", 
        type=str, 
        default=None,
        help="Staging file to process"
    )
    args = parser.parse_args()
    
    result = run_label_joiner(args.staging_file)
    
    if result.get("success"):
        print("\n" + "=" * 60)
        print("LABEL JOINER RESULT")
        print("=" * 60)
        print(f"Staging: {result['staging_file']}")
        print(f"Output: {result['output_file']}")
        print(f"Records: {result['records']}")
        print(f"Class 0: {result['class_0']}")
        print(f"Class 1: {result['class_1']}")
        print("=" * 60)
    else:
        print(f"Error: {result.get('error')}")