# Output for VPC
output "vpc_id" {
  value = aws_vpc.tech_demo_vpc.id
}

output "vpc_arn" {
  value = aws_vpc.tech_demo_vpc.arn
}

# Output for Internet Gateway
output "igw_id" {
  value = aws_internet_gateway.gw.id
}

output "igw_arn" {
  value = aws_internet_gateway.gw.arn
}

# Output for Subnets
output "public_subnet1_id" {
  value = aws_subnet.public1.id
}

output "public_subnet1_arn" {
  value = aws_subnet.public1.arn
}

output "public_subnet2_id" {
  value = aws_subnet.public2.id
}

output "public_subnet2_arn" {
  value = aws_subnet.public2.arn
}

output "private_subnet1_id" {
  value = aws_subnet.private1.id
}

output "private_subnet1_arn" {
  value = aws_subnet.private1.arn
}

output "private_subnet2_id" {
  value = aws_subnet.private2.id
}

output "private_subnet2_arn" {
  value = aws_subnet.private2.arn
}

/*
# Output for NAT Gateway
output "nat_gateway_id" {
  value = aws_nat_gateway.nat_gw.id
}
*/

# Output for Egress Only Internet Gateway
output "eoigw_id" {
  value = aws_egress_only_internet_gateway.eoigw.id
}

# Output for Security Group
/*
output "security_group_id" {
  value = aws_security_group.ec2_sg.id
}

output "security_group_arn" {
  value = aws_security_group.ec2_sg.arn
}
*/



