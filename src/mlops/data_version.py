import os
import logging
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "labeled")
DVC_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dvc")


class DataVersionManager:
    def __init__(
        self,
        data_dir: str = LABELED_DATA_DIR,
        dvc_dir: str = DVC_DATA_DIR,
        version_prefix: str = "v"
    ):
        self.data_dir = data_dir
        self.dvc_dir = dvc_dir
        self.version_prefix = version_prefix
        
        os.makedirs(dvc_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        self.version_file = os.path.join(dvc_dir, "versions.json")
        self.current_version = self._load_version()
    
    def _load_version(self) -> int:
        if os.path.exists(self.version_file):
            import json
            with open(self.version_file, 'r') as f:
                data = json.load(f)
                return data.get("current_version", 0)
        return 0
    
    def _save_version(self, version: int):
        import json
        with open(self.version_file, 'w') as f:
            json.dump({
                "current_version": version,
                "updated_at": datetime.now().isoformat()
            }, f)
    
    def get_current_version(self) -> str:
        return f"{self.version_prefix}{self.current_version:04d}"
    
    def get_latest_version_file(self) -> str:
        if not os.path.exists(self.data_dir):
            return ""
        
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.parquet')])
        
        if not files:
            return ""
        
        return os.path.join(self.data_dir, files[-1])
    
    def get_version_file(self, version: str) -> str:
        filepath = os.path.join(self.data_dir, f"labeled_batch_{version}.parquet")
        
        if os.path.exists(filepath):
            return filepath
        
        return os.path.join(self.data_dir, f"labeled_batch_{version}.parquet")
    
    def load_version(self, version: str = None) -> pd.DataFrame:
        if version is None:
            filepath = self.get_latest_version_file()
        else:
            filepath = self.get_version_file(version)
        
        if not filepath or not os.path.exists(filepath):
            logger.warning(f"Version file not found: {version}")
            return pd.DataFrame()
        
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded version {version or 'latest'}: {len(df)} records")
        
        return df
    
    def save_version(
        self, 
        df: pd.DataFrame, 
        version: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        if version is None:
            self.current_version += 1
            version = self.get_current_version()
        else:
            ver_num = int(version.replace(self.version_prefix, ''))
            if ver_num > self.current_version:
                self.current_version = ver_num
        
        filepath = os.path.join(self.data_dir, f"labeled_batch_{version}.parquet")
        
        df.to_parquet(filepath, index=False)
        
        self._save_version(self.current_version)
        
        metadata_file = os.path.join(self.dvc_dir, f"metadata_{version}.json")
        if metadata:
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved version {version}: {len(df)} records to {filepath}")
        
        return filepath
    
    def list_versions(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.data_dir):
            return []
        
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.parquet')])
        
        versions = []
        for f in files:
            version = f.replace("labeled_batch_", "").replace(".parquet", "")
            filepath = os.path.join(self.data_dir, f)
            
            stat = os.stat(filepath)
            
            versions.append({
                "version": version,
                "filepath": filepath,
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return versions
    
    def get_statistics(self, version: str = None) -> Dict[str, Any]:
        df = self.load_version(version)
        
        if df.empty:
            return {}
        
        stats = {
            "version": version or self.get_current_version(),
            "total_records": len(df),
            "columns": df.columns.tolist(),
        }
        
        if "Class" in df.columns:
            stats["class_distribution"] = {
                "normal": int((df["Class"] == 0).sum()),
                "fraud": int((df["Class"] == 1).sum()),
                "fraud_ratio": float((df["Class"] == 1).sum() / len(df))
            }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats["feature_stats"] = df[numeric_cols].describe().to_dict()
        
        return stats
    
    def rollback(self, target_version: str) -> bool:
        target_num = int(target_version.replace(self.version_prefix, ''))
        
        if target_num > self.current_version:
            logger.error(f"Version {target_version} does not exist")
            return False
        
        self.current_version = target_num
        self._save_version(self.current_version)
        
        logger.info(f"Rolled back to version {target_version}")
        return True


def run_dvc_manager():
    manager = DataVersionManager()
    
    print("\n=== Data Version Manager ===")
    print(f"Current version: {manager.get_current_version()}")
    
    versions = manager.list_versions()
    print(f"\nAvailable versions ({len(versions)}):")
    for v in versions:
        print(f"  {v['version']}: {v['size_bytes']} bytes, {v['modified_at']}")
    
    if versions:
        latest = versions[-1]
        stats = manager.get_statistics(latest['version'])
        if stats:
            print(f"\nStatistics for {latest['version']}:")
            print(f"  Total records: {stats['total_records']}")
            if 'class_distribution' in stats:
                print(f"  Normal: {stats['class_distribution']['normal']}")
                print(f"  Fraud: {stats['class_distribution']['fraud']}")


if __name__ == "__main__":
    run_dvc_manager()