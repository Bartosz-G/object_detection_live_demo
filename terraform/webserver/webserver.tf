data "local_file" "webserver_ssh_key_local" {
  filename = var.webserver_ssh_key_path
}

/*
resource "aws_secretsmanager_secret" "webserver-ssh-secret" {
  name        = var.webserver_ssh_secret_name
  description = "SSH Key to access webserver"
}

resource "aws_secretsmanager_secret_version" "ssh_key_version" {
  secret_id     = aws_secretsmanager_secret.webserver-ssh-secret.id
  secret_string = jsonencode({"ssh-key" : data.local_file.webserver_ssh_key_local.content})
}
*/

resource "aws_key_pair" "webserver-key-pair" {
  key_name   = "webserver_ssh_key_pair"
  public_key = file("${var.webserver_ssh_key_path}.pub")
}


resource "aws_iam_role" "webserver_role" {
  name = var.webserver_role_name

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

resource "aws_iam_policy" "webserver_iam" {
  name        = var.webserver_iam_name
  description = "Policy for Ansible master node to manage EC2 instances"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Effect = "Allow",
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "webserver_policy_attachment" {
  role       = aws_iam_role.webserver_role.name
  policy_arn = aws_iam_policy.webserver_iam.arn
}

resource "aws_iam_instance_profile" "webserver_profile" {
  name = "ansible_master_profile"
  role = aws_iam_role.webserver_role.name
}



resource "aws_instance" "webserver" {
  ami           = var.ami
  instance_type = var.instance_type

  iam_instance_profile = aws_iam_instance_profile.webserver_profile.name
  vpc_security_group_ids = [var.webserver_security_group]
  subnet_id = var.subnet_id
  key_name = aws_key_pair.webserver-key-pair.key_name
  associate_public_ip_address = true

      user_data = <<-EOF
                #!/bin/bash
                sudo apt-get update
                sudo apt-get install -y curl
              EOF

}



