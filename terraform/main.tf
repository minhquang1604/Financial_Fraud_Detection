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

module "vpc" {
  source = "./modules/vpc"
  name   = "mlops"
}

module "mlflow_sg" {
  source  = "./modules/sg"
  vpc_id  = module.vpc.vpc_id
  name    = "mlflow-server"
  ingress_rules = [
    { from_port = 5000, to_port = 5000, protocol = "tcp", description = "MLflow" },
    { from_port = 80,   to_port = 80,   protocol = "tcp", description = "HTTP" },
    { from_port = 443,  to_port = 443,  protocol = "tcp", description = "HTTPS" },
    { from_port = 22,   to_port = 22,   protocol = "tcp", description = "SSH" },
  ]
  egress_rules = [
    { from_port = 0, to_port = 0, protocol = "-1", description = "All outbound" },
  ]
}

module "mlflow_ec2" {
  source             = "./modules/ec2"
  ami                = var.ami
  instance_type      = var.instance_type
  subnet_id          = module.vpc.public_subnet_ids[0]
  key_name           = var.key_name
  name               = "mlflow-server"
  security_group_ids = [module.mlflow_sg.security_group_id]
}

module "db_sg" {
  source  = "./modules/sg"
  vpc_id  = module.vpc.vpc_id
  name    = "mlops-rds"
  ingress_rules = [
    { from_port = 5432, to_port = 5432, protocol = "tcp", cidr_ipv4 = module.vpc.vpc_cidr, description = "Postgres from VPC" },
  ]
  egress_rules = [
    { from_port = 0, to_port = 0, protocol = "-1", description = "All outbound" },
  ]
}

module "mlflow-backend-db_rds" {
  source             = "./modules/rds"
  name               = "mlops-fraud"
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.db_sg.security_group_id]
}

# locals {
#   images = ["api", "producer", "consumer", "drift-monitor", "webhook"]
# }

# module "ecr" {
#   source = "./modules/ecr"
#   for_each = toset(local.images)
#   name   = "mlops-${each.key}"
# }
