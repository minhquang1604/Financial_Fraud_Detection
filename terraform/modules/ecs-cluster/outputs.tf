output "cluster_id" {
  value = aws_ecs_cluster.this.id
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "security_group_id" {
  value = aws_security_group.this.id
}

output "service_connect_namespace_arn" {
  value = aws_service_discovery_http_namespace.this.arn
}
