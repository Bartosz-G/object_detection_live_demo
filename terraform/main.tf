terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    ansible = {
      version = "~> 1.1.0"
      source  = "ansible/ansible"
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

provider "ansible" {}

terraform {
  backend "s3" {
    bucket = "terraform-state-tech-demo"
    key    = "object-detection-demo/terraform.tfstate"
    region = "eu-west-2"
    dynamodb_table = "terraform-lock-tech-demo"
    encrypt = true
  }
}

# For initialising a backed on an s3
module "remote-backed" {
  source = "./remote_backend"

  state_bucket_name = "terraform-state-tech-demo"
  state_lock_db_name = "terraform-lock-tech-demo"
}


resource "ansible_group" "webservers" {
  name = "webservers"
}


module "network" {
  source = "./network"

  security_group_name = "tech_demo_security_group"
}

# TODO: Add WAF under the ALB for security (remember to route all traffic through ALB as not to access the EC2 itself)
# TODO: Improve security, limit ip adresses for SSH
resource "aws_security_group" "webserver_sg" {
  name        = "tech_demo_server_security_group"
  description = "[NOT FOR PROD] Security group for testing code on EC2 instance, improve safety before production"
  vpc_id      = module.network.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

    ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 1024
    to_port     = 65535
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

}






