output "repository_url" {
  description = "URL of the public ECR repository"
  value       = aws_ecrpublic_repository.this.repository_uri
}

output "registry_id" {
  description = "Registry ID"
  value       = aws_ecrpublic_repository.this.registry_id
}
