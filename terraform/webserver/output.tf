output "arn" {
  value = aws_instance.webserver.arn
}

output "id" {
  value = aws_instance.webserver.id
}

output "public_ip" {
  value = aws_instance.webserver.public_ip
}

output "webserver_ssh_key_path" {
  value = var.webserver_ssh_key_path
}