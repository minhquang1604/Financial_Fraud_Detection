terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

locals {
  name   = "mlops-prod"
  ecr    = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
  bucket = var.s3_bucket_name
  azs    = [for az in var.availability_zones : "${var.region}${az}"]
}

# ── Network ──────────────────────────────────
module "vpc" {
  source = "../../modules/vpc"
  name   = local.name

  availability_zones = local.azs
  nat_gateway_count  = 1
  public_subnets     = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets    = ["10.0.10.0/24", "10.0.11.0/24"]
}

# ── ECR Repos ────────────────────────────────
module "ecr" {
  source   = "../../modules/ecr"
  for_each = toset(["api", "producer", "consumer", "drift-monitor", "webhook", "kafka", "zookeeper"])
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

# ── ALB (public-facing, no routes yet) ────────
module "alb" {
  source = "../../modules/alb"

  name               = local.name
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  target_group_arns  = {}
  routes             = {}
}

# ── Outputs ──────────────────────────────────
output "vpc_id" {
  value = module.vpc.vpc_id
}

output "public_subnet_ids" {
  value = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  value = module.vpc.private_subnet_ids
}

output "vpc_cidr" {
  value = module.vpc.vpc_cidr
}

output "alb_listener_arn" {
  value = module.alb.listener_arn
}

output "alb_mlflow_listener_arn" {
  value = module.alb.mlflow_listener_arn
}

output "alb_dns_name" {
  value = module.alb.alb_dns_name
}

output "rds_address" {
  value = module.mlflow_db.db_address
}

output "rds_port" {
  value = module.mlflow_db.db_port
}

output "rds_db_name" {
  value = module.mlflow_db.db_name
}

output "ecr_repository_arns" {
  value = values(module.ecr)[*].repository_arn
}

output "ecr_registry" {
  value = local.ecr
}

output "ecs_cluster_id" {
  value = module.ecs_cluster.cluster_id
}

output "ecs_cluster_name" {
  value = module.ecs_cluster.cluster_name
}

output "execution_role_arn" {
  value = module.iam_ecs.execution_role_arn
}

output "execution_role_name" {
  value = module.iam_ecs.execution_role_name
}

output "task_role_arn" {
  value = module.iam_ecs.task_role_arn
}

output "task_role_name" {
  value = module.iam_ecs.task_role_name
}

output "ecs_security_group_id" {
  value = module.ecs_cluster.security_group_id
}

output "service_connect_namespace_arn" {
  value = module.ecs_cluster.service_connect_namespace_arn
}
