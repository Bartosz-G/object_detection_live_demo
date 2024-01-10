data "local_file" "ansible_master_ssh_key_local" {
  filename = var.ansible_master_ssh_key_path
}

resource "aws_secretsmanager_secret" "ansible-master-ssh-secret" {
  name        = var.ansible_master_ssh_secret_name
  description = "SSH Key to access ansible master node"
}

resource "aws_secretsmanager_secret_version" "ssh_key_version" {
  secret_id     = aws_secretsmanager_secret.ansible-master-ssh-secret.id
  secret_string = jsonencode({"ssh-key" : data.local_file.ansible_master_ssh_key_local.content})
}

resource "aws_key_pair" "ansible-master-key-pair" {
  key_name   = "ansible_master_ssh_key_pair"
  public_key = file(var.ansible_master_ssh_key_path)
}


resource "aws_iam_role" "ansible_master_role" {
  name = var.ansible_master_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
      },
    ]
  })
}

resource "aws_iam_policy" "ansible_master_iam" {
  name        = var.ansible_master_iam_name
  description = "Policy for Ansible master node to manage EC2 instances"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "ec2:DescribeInstances",
          "ec2:RebootInstances",
        ],
        Effect = "Allow",
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ansible_master_policy_attachment" {
  role       = aws_iam_role.ansible_master_role.name
  policy_arn = aws_iam_policy.ansible_master_iam.arn
}

resource "aws_iam_instance_profile" "ansible_master_profile" {
  name = "ansible_master_profile"
  role = aws_iam_role.ansible_master_role.name
}


resource "aws_security_group" "ansible_sg" {
  name        = var.ansible_security_group_name
  description = "[NOT FOR PROD] Security group for testing code on EC2 instance, improve safety before production"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

}

resource "aws_instance" "ansible_master_node" {
  ami           = "ami-00efc25778562c229"
  instance_type = "t4g.micro"

  iam_instance_profile = aws_iam_instance_profile.ansible_master_profile.name
  vpc_security_group_ids = [aws_security_group.ansible_sg.id]
  subnet_id = var.subnet_id
  key_name = aws_key_pair.ansible-master-key-pair.key_name
  associate_public_ip_address = true

    user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update -y
              sudo apt-get install software-properties-common -y
              sudo add-apt-repository --yes --update ppa:ansible/ansible
              sudo apt-get install ansible -y
              EOF
}



