import os
import io
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class S3DataManager:
    BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "fraud-detection-data")
    REGION = os.environ.get("AWS_REGION", "us-east-1")

    PREFIXES = {
        "realtime": "data/realtime/",
        "labeled": "data/labeled/",
        "processed": "data/processed/",
        "mixed": "data/mixed/",
        "train": "data/train/",
    }

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None
    ):
        self.bucket_name = bucket_name or self.BUCKET_NAME
        self.region_name = region_name or self.REGION

        session_kwargs = {"region_name": self.region_name}

        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.s3 = boto3.client("s3", **session_kwargs)
        self.resource = boto3.resource("s3", **session_kwargs)

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' exists")
        except ClientError:
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created bucket '{self.bucket_name}'")
            except ClientError as e:
                logger.warning(f"Cannot create bucket: {e}")

    def _get_s3_path(self, prefix_type: str, filename: str) -> str:
        return f"{self.PREFIXES.get(prefix_type, '')}{filename}"

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        prefix_type: str,
        filename: Optional[str] = None,
        format: str = "parquet"
    ) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_{timestamp}.{format}"

        s3_path = self._get_s3_path(prefix_type, filename)

        buffer = io.BytesIO()

        if format == "parquet":
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            content_type = "application/octet-stream"
        else:
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            content_type = "text/csv"

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_path,
            Body=buffer.getvalue(),
            ContentType=content_type
        )

        logger.info(f"Uploaded to s3://{self.bucket_name}/{s3_path}")
        return s3_path

    def download_dataframe(
        self,
        s3_path: str,
        format: str = "parquet"
    ) -> pd.DataFrame:
        response = self.s3.get_object(
            Bucket=self.bucket_name,
            Key=s3_path
        )

        body = response["Body"].read()

        if format == "parquet":
            return pd.read_parquet(io.BytesIO(body))
        else:
            return pd.read_csv(io.BytesIO(body))

    def list_files(self, prefix_type: str) -> List[str]:
        prefix = self.PREFIXES.get(prefix_type, "")
        if not prefix:
            return []

        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )

        files = []
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                if not key.endswith("/"):
                    files.append(key)

        return sorted(files, key=lambda x: x.split("/")[-1])

    def download_latest_files(
        self,
        prefix_type: str,
        n: int = 5
    ) -> List[pd.DataFrame]:
        files = self.list_files(prefix_type)
        if not files:
            logger.warning(f"No files found for prefix: {prefix_type}")
            return []

        recent_files = files[-n:]
        dfs = []

        for s3_path in recent_files:
            try:
                format = "csv" if s3_path.endswith(".csv") else "parquet"
                df = self.download_dataframe(s3_path, format=format)
                dfs.append(df)
                logger.info(f"Downloaded {s3_path}")
            except Exception as e:
                logger.error(f"Failed to download {s3_path}: {e}")

        return dfs

    def delete_file(self, s3_path: str):
        self.s3.delete_object(
            Bucket=self.bucket_name,
            Key=s3_path
        )
        logger.info(f"Deleted s3://{self.bucket_name}/{s3_path}")

    def file_exists(self, s3_path: str) -> bool:
        try:
            self.s3.head_object(
                Bucket=self.bucket_name,
                Key=s3_path
            )
            return True
        except ClientError:
            return False

    def get_file_url(self, s3_path: str, expiration: int = 3600) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": s3_path
            },
            ExpiresIn=expiration
        )

    def upload_json(
        self,
        data: Dict[str, Any],
        prefix_type: str,
        filename: str
    ) -> str:
        s3_path = self._get_s3_path(prefix_type, filename)

        content = json.dumps(data, indent=2)

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_path,
            Body=content.encode("utf-8"),
            ContentType="application/json"
        )

        logger.info(f"Uploaded JSON to s3://{self.bucket_name}/{s3_path}")
        return s3_path

    def download_json(self, s3_path: str) -> Dict[str, Any]:
        response = self.s3.get_object(
            Bucket=self.bucket_name,
            Key=s3_path
        )
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)


def get_s3_manager() -> S3DataManager:
    return S3DataManager()