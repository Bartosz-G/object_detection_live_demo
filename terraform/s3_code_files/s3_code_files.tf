resource "aws_s3_bucket" "s3_code_files" {
  bucket = var.bucket_name
}

resource "aws_s3_object" "code_files_object_put" {
  for_each = fileset(var.local_files_path, "**/*")

  bucket = aws_s3_bucket.s3_code_files.id
  key    = each.value
  source = "${var.local_files_path}/${each.value}"
}