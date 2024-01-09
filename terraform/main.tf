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