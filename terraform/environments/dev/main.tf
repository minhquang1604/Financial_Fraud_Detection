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
  backend "s3" {
    bucket = "fraud-tfstate"
    key    = "dev/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"
}

locals {
  name   = "mlops-dev"
  images = ["api", "producer", "consumer", "drift-monitor", "webhook"]
}

# ── Network ──────────────────────────────────
module "vpc" {
  source = "../../modules/vpc"
  name   = local.name

  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = []
}

# ── Security Groups ──────────────────────────
module "api_sg" {
  source = "../../modules/sg"
  vpc_id = module.vpc.vpc_id
  name   = "${local.name}-api"

  ingress_rules = [
    { from_port = 8000, to_port = 8000, protocol = "tcp", description = "FastAPI" },
  ]
  egress_rules = [{ from_port = 0, to_port = 0, protocol = "-1" }]
}

module "kafka_sg" {
  source = "../../modules/sg"
  vpc_id = module.vpc.vpc_id
  name   = "${local.name}-kafka"

  ingress_rules = [
    { from_port = 9092, to_port = 9092, protocol = "tcp", description = "Kafka" },
    { from_port = 2181, to_port = 2181, protocol = "tcp", description = "Zookeeper" },
  ]
  egress_rules = [{ from_port = 0, to_port = 0, protocol = "-1" }]
}

module "monitoring_sg" {
  source = "../../modules/sg"
  vpc_id = module.vpc.vpc_id
  name   = "${local.name}-monitoring"

  ingress_rules = [
    { from_port = 9090, to_port = 9090, protocol = "tcp", description = "Prometheus" },
    { from_port = 3000, to_port = 3000, protocol = "tcp", description = "Grafana" },
  ]
  egress_rules = [{ from_port = 0, to_port = 0, protocol = "-1" }]
}

# ── ECR Repos ────────────────────────────────
module "ecr_public" {
  source = "../../modules/ecr-public"
  for_each = toset(local.images)
  name   = "${local.name}-${each.key}"
}
