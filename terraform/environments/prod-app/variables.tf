variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

variable "image_tag" {
  description = "Image tag for custom services"
  type        = string
  default     = "latest"
}

variable "s3_bucket_name" {
  description = "S3 bucket for streaming data, training data, models"
  type        = string
}

variable "mlflow_artifact_bucket" {
  description = "S3 bucket for MLflow artifacts (separate from data bucket)"
  type        = string
  default     = "fraud-detection-uit-mlflow-artifacts"
}

variable "rds_username" {
  description = "Master username for MLflow RDS database"
  type        = string
}

variable "rds_password" {
  description = "Master password for MLflow RDS database"
  type        = string
  sensitive   = true
}
