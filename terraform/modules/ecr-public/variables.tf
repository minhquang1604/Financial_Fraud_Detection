variable "name" {
  description = "Name of the public ECR repository"
  type        = string
}

variable "catalog_data" {
  description = "Catalog data for the public repository"
  type = object({
    about_text        = optional(string)
    architectures     = optional(list(string))
    operating_systems = optional(list(string))
    usage_text        = optional(string)
  })
  default = {}
}
