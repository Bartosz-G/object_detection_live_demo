terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "eu-west-2"
    default_tags {
      tags = {
        Environment     = "Dev"
        Service         = "object-detection-live-demo"
        HashiCorp-Learn = "aws-default-tags"
      }
    }
}


module "remote-backed" {
  source = "./remote_backend"

  state_bucket_name = "terraform-state-tech-demo"
  state_lock_db_name = "terraform-lock-tech-demo"
}

