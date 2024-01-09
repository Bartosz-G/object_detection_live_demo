output "state_bucket_id" {
  value = aws_s3_bucket.terraform_state_bucket.id
}

output "state_bucket_arn" {
  value = aws_s3_bucket.terraform_state_bucket.arn
}

output "terraform_lock_id" {
  value = aws_dynamodb_table.terraform_state_lock.id
}

output "terraform_lock_arn" {
  value = aws_dynamodb_table.terraform_state_lock.arn
}