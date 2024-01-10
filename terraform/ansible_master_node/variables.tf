variable "vpc_id" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "ansible_security_group_name" {
  type = string
  default = "ansible_sg"
}

variable "ansible_master_ssh_key_path" {
  type = string
}

variable "ansible_master_ssh_secret_name" {
  type = string
  default = "ansible_master_ssh_key"
}

variable "ansible_master_role_name" {
  type = string
  default = "ansible_master_role"
}

variable "ansible_master_iam_name" {
  type = string
  default = "ansible_master_iam"
}

variable "ansible_master_instance_name" {
  type = string
  default = "ansible_master"
}