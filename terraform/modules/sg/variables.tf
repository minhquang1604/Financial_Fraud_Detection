variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "name" {
  description = "Name tag"
  type        = string
}

variable "description" {
  description = "Description of the security group"
  type        = string
  default     = "Managed by Terraform"
}

variable "ingress_rules" {
  description = "List of ingress rules"
  type = list(object({
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_ipv4   = optional(string, "0.0.0.0/0")
    description = optional(string, "")
  }))
  default = []
}

variable "egress_rules" {
  description = "List of egress rules"
  type = list(object({
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_ipv4   = optional(string, "0.0.0.0/0")
    description = optional(string, "")
  }))
  default = [
    {
      from_port = 0
      to_port   = 0
      protocol  = "-1"
    }
  ]
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
