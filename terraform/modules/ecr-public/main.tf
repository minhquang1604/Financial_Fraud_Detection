terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

resource "aws_ecrpublic_repository" "this" {
  repository_name = var.name

  dynamic "catalog_data" {
    for_each = length(keys(var.catalog_data)) > 0 ? [var.catalog_data] : []
    content {
      about_text        = lookup(catalog_data.value, "about_text", null)
      architectures     = lookup(catalog_data.value, "architectures", null)
      operating_systems = lookup(catalog_data.value, "operating_systems", null)
      usage_text        = lookup(catalog_data.value, "usage_text", null)
    }
  }
}
