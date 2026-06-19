terraform {
  backend "s3" {
    bucket = "aws-terraform-remotebackend"
    key    = "tfstate/mlops-project/app.tfstate"
    region = "ap-southeast-1"
  }
}
