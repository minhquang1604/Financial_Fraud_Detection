terraform {
  backend "s3" {
    bucket = "aws-terraform-remotebackend"
    key    = "tfstate/mlops-project/infra.tfstate"
    region = "ap-southeast-1"
  }
}
