variable "name" {
  description = "Prefix for all resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
}

variable "cluster_id" {
  description = "ECS cluster ID"
  type        = string
}

variable "execution_role_arn" {
  description = "ECS execution role ARN"
  type        = string
}

variable "task_role_arn" {
  description = "ECS task role ARN"
  type        = string
}

variable "security_group_id" {
  description = "ECS tasks security group ID"
  type        = string
}

variable "assign_public_ip" {
  description = "Assign public IP to ECS tasks"
  type        = bool
  default     = false
}

variable "efs_filesystem_id" {
  description = "EFS filesystem ID for persistent volumes (optional)"
  type        = string
  default     = null
}

variable "service_connect_namespace_arn" {
  description = "ARN of the Service Connect namespace"
  type        = string
}

variable "services" {
  description = "Map of service configurations"
  type = map(object({
    image        = string
    cpu          = number
    memory       = number
    port         = optional(number)
    desired_count = optional(number, 1)
    is_service   = optional(bool, true)
    use_spot     = optional(bool, false)
    register_with_lb = optional(bool, true)
    assign_public_ip = optional(bool)
    enable_service_connect = optional(bool, false)
    subnet_ids   = optional(list(string))
    health_check_path = optional(string)
    health_check_matcher = optional(string)
    environment  = optional(map(string), {})
    command      = optional(list(string), [])
    mount_path   = optional(string)
  }))
}
