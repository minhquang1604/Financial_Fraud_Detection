variable "name" {
  description = "Prefix for all resources"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for the ALB security group"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for the ALB"
  type        = list(string)
}

variable "target_group_arns" {
  description = "Map of service name to target group ARN"
  type        = map(string)
}

variable "routes" {
  description = "ALB listener route configurations"
  type = map(object({
    path_patterns = list(string)
    priority      = number
  }))
}
