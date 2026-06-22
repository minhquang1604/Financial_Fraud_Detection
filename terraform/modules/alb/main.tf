resource "aws_security_group" "this" {
  name        = "${var.name}-alb"
  description = "ALB public"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name}-alb" }
}

resource "aws_lb" "this" {
  name               = var.name
  internal           = false
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  security_groups    = [aws_security_group.this.id]
  tags               = { Name = var.name }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "Fraud Detection API — use /predict, /health, /docs, /model/info"
      status_code  = "200"
    }
  }
}

resource "aws_lb_listener" "mlflow" {
  load_balancer_arn = aws_lb.this.arn
  port              = 5000
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "MLflow endpoint"
      status_code  = "200"
    }
  }
}

resource "aws_lb_listener_rule" "this" {
  for_each = var.routes

  listener_arn = aws_lb_listener.http.arn
  priority     = each.value.priority

  action {
    type             = "forward"
    target_group_arn = var.target_group_arns[each.key]
  }

  condition {
    path_pattern { values = each.value.path_patterns }
  }
}
