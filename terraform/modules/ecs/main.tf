# ── Log Groups ───────────────────────────────
resource "aws_cloudwatch_log_group" "this" {
  for_each = var.services
  name              = "/ecs/${var.name}-${each.key}"
  retention_in_days = 7
}

data "aws_region" "current" {}

# ── Container Definitions ────────────────────
locals {
  container_defs = {
    for name, svc in var.services : name => jsonencode([
      {
        name      = "${var.name}-${name}"
        image     = svc.image
        essential = true
        command   = length(svc.command) > 0 ? svc.command : null
        portMappings = svc.port != null ? [
          {
            containerPort = svc.port
            hostPort      = svc.port
            protocol      = "tcp"
            name          = name
          }
        ] : []
        environment = [
          for k, v in svc.environment : {
            name  = k
            value = v
          }
        ]
        mountPoints = svc.mount_path != null && var.efs_filesystem_id != null ? [
          {
            sourceVolume  = "efs-${name}"
            containerPath = svc.mount_path
            readOnly      = false
          }
        ] : []
        logConfiguration = {
          logDriver = "awslogs"
          options = {
            "awslogs-group"         = "/ecs/${var.name}-${name}"
            "awslogs-region"        = data.aws_region.current.name
            "awslogs-stream-prefix" = "ecs"
          }
        }
      }
    ])
  }
}

# ── Task Definitions ─────────────────────────
resource "aws_ecs_task_definition" "this" {
  for_each = var.services

  family                   = "${var.name}-${each.key}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn
  container_definitions    = local.container_defs[each.key]

  dynamic "volume" {
    for_each = each.value.mount_path != null && var.efs_filesystem_id != null ? [1] : []
    content {
      name = "efs-${each.key}"
      efs_volume_configuration {
        file_system_id     = var.efs_filesystem_id
        root_directory     = "/${each.key}"
        transit_encryption = "ENABLED"
      }
    }
  }
}

# ── Target Groups (for port-exposed services) ─
resource "aws_lb_target_group" "this" {
  for_each = {
    for name, svc in var.services : name => svc
    if svc.port != null && svc.is_service && svc.register_with_lb
  }

  name        = "${var.name}-${each.key}"
  port        = each.value.port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    path                = each.value.health_check_path != null ? each.value.health_check_path : "/"
    port                = "traffic-port"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = each.value.health_check_matcher != null ? each.value.health_check_matcher : "200"
  }

  tags = { Name = "${var.name}-${each.key}" }
}

# ── ECS Services (long-running) ──────────────
resource "aws_ecs_service" "this" {
  for_each = {
    for name, svc in var.services : name => svc
    if svc.is_service
  }

  name            = "${var.name}-${each.key}"
  cluster         = var.cluster_id
  task_definition = aws_ecs_task_definition.this[each.key].arn
  desired_count   = each.value.desired_count

  capacity_provider_strategy {
    capacity_provider = each.value.use_spot ? "FARGATE_SPOT" : "FARGATE"
    weight            = 1
    base              = each.value.use_spot ? null : 1
  }

  dynamic "service_connect_configuration" {
    for_each = each.value.port != null && each.value.enable_service_connect ? [1] : []
    content {
      enabled   = true
      namespace = var.service_connect_namespace_arn
      service {
        port_name = each.key
        client_alias {
          dns_name = "${var.name}-${each.key}"
          port     = each.value.port
        }
      }
    }
  }

  dynamic "load_balancer" {
    for_each = each.value.port != null && each.value.register_with_lb ? [1] : []
    content {
      container_name   = "${var.name}-${each.key}"
      container_port   = each.value.port
      target_group_arn = aws_lb_target_group.this[each.key].arn
    }
  }

  network_configuration {
    subnets          = each.value.subnet_ids != null ? each.value.subnet_ids : var.subnet_ids
    assign_public_ip = each.value.assign_public_ip != null ? each.value.assign_public_ip : var.assign_public_ip
    security_groups  = [var.security_group_id]
  }
}
