output "task_definition_arns" {
  value = {
    for name, td in aws_ecs_task_definition.this : name => td.arn
  }
}

output "target_group_arns" {
  value = {
    for name, tg in aws_lb_target_group.this : name => tg.arn
  }
}

output "service_names" {
  value = {
    for name, svc in aws_ecs_service.this : name => svc.name
  }
}
