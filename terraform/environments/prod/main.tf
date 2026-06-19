terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  name       = "mlops-prod"
  img        = var.image_tag
  ecr        = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
  bucket     = var.s3_bucket_name
  mlflow_uri = "http://mlops-prod-mlflow:5000"
  kafka      = "mlops-prod-kafka:9092"
  azs        = [for az in var.availability_zones : "${var.region}${az}"]
}

# ── Network ──────────────────────────────────
module "vpc" {
  source = "../../modules/vpc"
  name   = local.name

  availability_zones = local.azs
  public_subnets     = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets    = ["10.0.10.0/24", "10.0.11.0/24"]
}

# ── ECR Repos ────────────────────────────────
module "ecr" {
  source   = "../../modules/ecr"
  for_each = toset(["api", "producer", "consumer", "drift-monitor", "webhook"])
  name     = "${local.name}-${each.key}"
}

# ── RDS (MLflow backend) ─────────────────────
module "mlflow_db_sg" {
  source = "../../modules/sg"
  vpc_id = module.vpc.vpc_id
  name   = "${local.name}-mlflow-db"
  ingress_rules = [
    { from_port = 5432, to_port = 5432, protocol = "tcp", cidr_ipv4 = module.vpc.vpc_cidr, description = "Postgres from VPC" },
  ]
  egress_rules = [
    { from_port = 0, to_port = 0, protocol = "-1", description = "All outbound" },
  ]
}

module "mlflow_db" {
  source             = "../../modules/rds"
  name               = "${local.name}-mlflow"
  engine             = "postgres"
  engine_version     = "15.18"
  db_name            = "mlflow"
  username           = var.rds_username
  password           = var.rds_password
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.mlflow_db_sg.security_group_id]
  skip_final_snapshot = true
}

# ── ECS Cluster Infra ────────────────────────
module "ecs_cluster" {
  source = "../../modules/ecs-cluster"
  name   = local.name
  vpc_id = module.vpc.vpc_id
}

module "iam_ecs" {
  source = "../../modules/iam-ecs"
  name   = local.name
}

# ── ECS Services ─────────────────────────────
module "ecs" {
  source = "../../modules/ecs"
  name   = local.name
  vpc_id = module.vpc.vpc_id
  subnet_ids       = module.vpc.private_subnet_ids
  cluster_id         = module.ecs_cluster.cluster_id
  execution_role_arn = module.iam_ecs.execution_role_arn
  task_role_arn      = module.iam_ecs.task_role_arn
  security_group_id  = module.ecs_cluster.security_group_id

  services = {
    # ── MLflow (model registry) ──
    mlflow = {
      image           = "ghcr.io/mlflow/mlflow:v2.14.0"
      cpu             = 256
      memory          = 512
      port            = 5000
      desired_count   = 1
      health_check_path = "/health"
      environment = {
        MLFLOW_ARTIFACT_ROOT = "s3://${local.bucket}/mlflow/"
      }
      command = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", "postgresql://${var.rds_username}:${var.rds_password}@${module.mlflow_db.db_address}:${module.mlflow_db.db_port}/mlflow",
        "--default-artifact-root", "s3://${local.bucket}/mlflow/",
        "--artifacts-destination", "s3://${local.bucket}/mlflow/"
      ]
    }

    # ── FastAPI prediction service ──
    api = {
      image        = "${local.ecr}/${local.name}-api:${local.img}"
      cpu          = 512
      memory       = 1024
      port         = 8000
      desired_count = 1
      health_check_path = "/health"
      environment = {
        MLFLOW_TRACKING_URI = local.mlflow_uri
        USE_S3              = "true"
        S3_BUCKET_NAME      = local.bucket
      }
    }

    # ── Kafka Zookeeper ──
    zookeeper = {
      image         = "confluentinc/cp-zookeeper:7.5.0"
      cpu           = 256
      memory        = 512
      desired_count = 1
      use_spot      = true
      environment   = { ZOOKEEPER_CLIENT_PORT = "2181", ZOOKEEPER_TICK_TIME = "2000" }
    }

    # ── Kafka broker ──
    kafka = {
      image         = "confluentinc/cp-kafka:7.5.0"
      cpu           = 512
      memory        = 1024
      desired_count = 1
      environment = {
        KAFKA_BROKER_ID                          = "1"
        KAFKA_ZOOKEEPER_CONNECT                  = "mlops-prod-zookeeper:2181"
        KAFKA_ADVERTISED_LISTENERS               = "PLAINTEXT://mlops-prod-kafka:9092"
        KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR   = "1"
        KAFKA_LOG_RETENTION_HOURS                = "168"
        KAFKA_AUTO_CREATE_TOPICS_ENABLE          = "true"
      }
    }

    # ── Kafka consumer ──
    consumer = {
      image         = "${local.ecr}/${local.name}-consumer:${local.img}"
      cpu           = 256
      memory        = 512
      desired_count = 1
      use_spot      = true
      environment = {
        KAFKA_BOOTSTRAP_SERVERS = local.kafka
        API_URL                 = "http://mlops-prod-api:8000"
        USE_S3                  = "true"
        S3_BUCKET_NAME          = local.bucket
      }
    }

    # ── Producer (RunTask job) ──
    producer = {
      image       = "${local.ecr}/${local.name}-producer:${local.img}"
      cpu         = 256
      memory      = 512
      is_service  = false
      environment = { KAFKA_BOOTSTRAP_SERVERS = local.kafka }
    }

    # ── Drift monitor (RunTask job) ──
    drift_monitor = {
      image       = "${local.ecr}/${local.name}-drift-monitor:${local.img}"
      cpu         = 256
      memory      = 512
      is_service  = false
      environment = { MLFLOW_TRACKING_URI = local.mlflow_uri, S3_BUCKET_NAME = local.bucket }
    }

    # ── Prometheus ──
    prometheus = {
      image         = "prom/prometheus:v2.47.0"
      cpu           = 256
      memory        = 512
      port          = 9090
      desired_count = 1
      use_spot      = true
      command = [
        "--config.file=/etc/prometheus/prometheus.yml",
        "--storage.tsdb.path=/prometheus",
        "--storage.tsdb.retention.time=15d"
      ]
    }

    # ── Grafana ──
    grafana = {
      image         = "grafana/grafana:10.1.0"
      cpu           = 256
      memory        = 512
      port          = 3000
      desired_count = 1
      use_spot      = true
      environment   = { GF_SECURITY_ADMIN_USER = "admin", GF_SECURITY_ADMIN_PASSWORD = "admin" }
    }

    # ── Webhook receiver ──
    webhook = {
      image         = "${local.ecr}/${local.name}-webhook:${local.img}"
      cpu           = 256
      memory        = 512
      desired_count = 1
      use_spot      = true
    }
  }
}

# ── ALB (public-facing) ──────────────────────
module "alb" {
  source = "../../modules/alb"

  name               = local.name
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  target_group_arns  = module.ecs.target_group_arns

  routes = {
    api = {
      path_patterns = ["/api/*", "/predict", "/health", "/metrics", "/model/*"]
      priority      = 100
    }
    grafana = {
      path_patterns = ["/grafana/*"]
      priority      = 110
    }
    prometheus = {
      path_patterns = ["/prometheus/*"]
      priority      = 120
    }
    mlflow = {
      path_patterns = ["/mlflow/*"]
      priority      = 130
    }
  }
}

# ── Outputs ──────────────────────────────────
output "alb_dns" {
  value = module.alb.alb_dns_name
}

output "cluster_name" {
  value = module.ecs_cluster.cluster_name
}

output "mlflow_uri" {
  value = local.mlflow_uri
}

output "rds_address" {
  value = module.mlflow_db.db_address
}
