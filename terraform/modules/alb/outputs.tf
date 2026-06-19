output "alb_dns_name" {
  value = aws_lb.this.dns_name
}

output "alb_arn" {
  value = aws_lb.this.arn
}

output "alb_sg_id" {
  value = aws_security_group.this.id
}

output "listener_arn" {
  value = aws_lb_listener.http.arn
}

output "mlflow_listener_arn" {
  value = aws_lb_listener.mlflow.arn
}
