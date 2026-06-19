variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

variable "availability_zones" {
  description = "List of availability zones for the VPC"
  type        = list(string)
  default     = ["a", "b"]
}

variable "s3_bucket_name" {
  description = "S3 bucket for data, models, MLflow artifacts"
  type        = string
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
