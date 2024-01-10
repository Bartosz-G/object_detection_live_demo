variable "vpc_id" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "webserver_ssh_key_path" {
  type = string
}

variable "ami" {
  type = string
  default = "ami-00efc25778562c229" //Ubuntu
}

variable "instance_type" {
  type = string
  default = "t4g.micro" //Newest burstable instances
}

variable "webserver_security_group_name" {
  type = string
  default = "tech_demo_sg"
}

variable "webserver_ssh_secret_name" {
  type = string
  default = "tech_demo_ssh_key"
}

variable "webserver_role_name" {
  type = string
  default = "tech_demo_role"
}

variable "webserver_iam_name" {
  type = string
  default = "tech_demo_iam"
}

variable "webserver_instance_name" {
  type = string
  default = "tech_demo_master"
}
