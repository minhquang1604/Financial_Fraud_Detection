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

data "terraform_remote_state" "infra" {
  backend = "s3"
  config = {
    bucket = "aws-terraform-remotebackend"
    key    = "tfstate/mlops-project/infra.tfstate"
    region = "ap-southeast-1"
  }
}

data "aws_caller_identity" "current" {}

locals {
  name       = "mlops-prod"
  img        = var.image_tag
  ecr        = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
  bucket     = var.s3_bucket_name
  mlflow_artifact_bucket = var.mlflow_artifact_bucket
  mlflow_uri = "http://mlops-prod-mlflow:5000"
  kafka      = "mlops-prod-kafka-kafka:9092"

  infra = data.terraform_remote_state.infra.outputs
}

# ── ECS Cluster + Services ───────────────────
module "ecs" {
  source = "../../modules/ecs"
  name   = local.name
  vpc_id = local.infra.vpc_id
  subnet_ids = local.infra.private_subnet_ids
  cluster_id         = local.infra.ecs_cluster_id
  execution_role_arn = local.infra.execution_role_arn
  task_role_arn      = local.infra.task_role_arn
  security_group_id        = local.infra.ecs_security_group_id
  service_connect_namespace_arn = local.infra.service_connect_namespace_arn

  services = {
    # ── MLflow (model registry) ──
    mlflow = {
      image           = "ghcr.io/mlflow/mlflow:v2.17.0"
      cpu             = 512
      memory          = 1024
      port            = 5000
      desired_count   = 1
      health_check_path = "/"
      health_check_matcher = "200-399"
      enable_service_connect = true
      environment = {
        MLFLOW_ARTIFACT_ROOT = "s3://${local.mlflow_artifact_bucket}/"
      }
      command = [
        "sh", "-c",
        "pip install psycopg2-binary boto3 -q && mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://${var.rds_username}:${var.rds_password}@${local.infra.rds_address}:${local.infra.rds_port}/mlflow --default-artifact-root s3://${local.mlflow_artifact_bucket}/ --artifacts-destination s3://${local.mlflow_artifact_bucket}/"
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
      enable_service_connect = true
      environment = {
        MLFLOW_TRACKING_URI  = local.mlflow_uri
        USE_S3               = "true"
        S3_BUCKET_NAME       = local.bucket
        MODEL_POLL_INTERVAL  = "15"
      }
    }

  }
}



# ── 2. PUBLIC SERVICES (Kafka Broker) ──
module "ecs_kafka" {
  source     = "../../modules/ecs"
  name       = "${local.name}-kafka"      # Given a unique name to avoid resource conflicts
  vpc_id     = local.infra.vpc_id
  subnet_ids = local.infra.public_subnet_ids # Moved to Public Subnets!
  
  cluster_id                    = local.infra.ecs_cluster_id
  execution_role_arn            = local.infra.execution_role_arn
  task_role_arn                 = local.infra.task_role_arn
  security_group_id             = local.infra.ecs_security_group_id
  service_connect_namespace_arn = local.infra.service_connect_namespace_arn

  services = {
    kafka = {
      image                  = "apache/kafka:4.2.1"
      cpu                    = 1024
      memory                 = 2048
      port                   = 9092
      desired_count          = 1
      register_with_lb      = false
      assign_public_ip       = true
      enable_service_connect = false

      command = [
        "bash", "-c",
        <<-SH
	IP="$(hostname -i)"
	PUBLIC_IP="$(wget -q -O - http://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || echo "$IP")"
	CID="5976314f-fc61-439b-9fcb-b7990714650f"
	mkdir -p /tmp/kafka-logs
	MK=/tmp/kafka.properties
	cat > $MK <<PROPS
	process.roles=broker,controller
	node.id=1
	listeners=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
	advertised.listeners=PLAINTEXT://$PUBLIC_IP:9092
	controller.listener.names=CONTROLLER
	listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
	controller.quorum.voters=1@127.0.0.1:9093
	offsets.topic.replication.factor=1
	auto.create.topics.enable=true
	log.dirs=/tmp/kafka-logs
	num.partitions=3
	PROPS
	[ ! -f /tmp/kafka-logs/meta.properties ] && /opt/kafka/bin/kafka-storage.sh format -t "$CID" -c "$MK" >/dev/null 2>&1 || true
	exec /opt/kafka/bin/kafka-server-start.sh "$MK"
	SH
      ]
    }
  }
}















# ── ALB Listener Rules ──────────────────────
resource "aws_lb_listener_rule" "api" {
  listener_arn = local.infra.alb_listener_arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = module.ecs.target_group_arns["api"]
  }

  condition {
    path_pattern { values = ["/api/*", "/predict", "/health", "/metrics", "/model/*"] }
  }
}

# resource "aws_lb_listener_rule" "grafana" {
#   listener_arn = local.infra.alb_listener_arn
#   priority     = 110

#   action {
#     type             = "forward"
#     target_group_arn = module.ecs.target_group_arns["grafana"]
#   }

#   condition {
#     path_pattern { values = ["/grafana/*"] }
#   }
# }

# resource "aws_lb_listener_rule" "prometheus" {
#   listener_arn = local.infra.alb_listener_arn
#   priority     = 120

#   action {
#     type             = "forward"
#     target_group_arn = module.ecs.target_group_arns["prometheus"]
#   }

#   condition {
#     path_pattern { values = ["/prometheus/*"] }
#   }
# }

resource "aws_lb_listener_rule" "mlflow" {
  listener_arn = local.infra.alb_listener_arn
  priority     = 130

  action {
    type             = "forward"
    target_group_arn = module.ecs.target_group_arns["mlflow"]
  }

  condition {
    path_pattern { values = ["/mlflow/*"] }
  }
}

resource "aws_lb_listener_rule" "mlflow_direct" {
  listener_arn = local.infra.alb_mlflow_listener_arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = module.ecs.target_group_arns["mlflow"]
  }

  condition {
    path_pattern { values = ["/*"] }
  }
}

# ── Outputs ──────────────────────────────────
output "cluster_name" {
  value = local.infra.ecs_cluster_name
}

output "mlflow_uri" {
  value = local.mlflow_uri
}

output "alb_dns" {
  value = local.infra.alb_dns_name
}
