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

terraform {
  backend "s3" {
    bucket = "terraform-state-tech-demo"
    key    = "object-detection-demo/terraform.tfstate"
    region = "eu-west-2"
    dynamodb_table = "terraform-lock-tech-demo"
    encrypt = true
  }
}


module "remote-backed" {
  source = "./remote_backend"

  state_bucket_name = "terraform-state-tech-demo"
  state_lock_db_name = "terraform-lock-tech-demo"
}


module "network" {
  source = "./network"

  security_group_name = "tech_demo_security_group"
}

# TODO: Add WAF under the ALB for security (remember to route all traffic through ALB as not to access the EC2 itself)




