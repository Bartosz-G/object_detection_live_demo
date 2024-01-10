# TODO: Add traffic logging for VPC through VPC Flow

resource "aws_vpc" "tech_demo_vpc" {
  cidr_block = "10.0.0.0/16"

  assign_generated_ipv6_cidr_block = true

  enable_dns_support = true
  enable_dns_hostnames = true
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.tech_demo_vpc.id
}

# Public Subnets
resource "aws_subnet" "public1" {
  vpc_id     = aws_vpc.tech_demo_vpc.id
  cidr_block = "10.0.0.0/24"

  availability_zone = "eu-west-2b"

  map_public_ip_on_launch = true
  assign_ipv6_address_on_creation = true
  ipv6_cidr_block = cidrsubnet(aws_vpc.tech_demo_vpc.ipv6_cidr_block, 8, 1)
}

resource "aws_subnet" "public2" {
  vpc_id     = aws_vpc.tech_demo_vpc.id
  cidr_block = "10.0.1.0/24"

  availability_zone = "eu-west-2c"

  map_public_ip_on_launch = true
  assign_ipv6_address_on_creation = true
  ipv6_cidr_block = cidrsubnet(aws_vpc.tech_demo_vpc.ipv6_cidr_block, 8, 2)
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.tech_demo_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }

  route {
    ipv6_cidr_block = "::/0"
    gateway_id = aws_internet_gateway.gw.id
  }
}

resource "aws_route_table_association" "public1" {
  subnet_id      = aws_subnet.public1.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public2" {
  subnet_id      = aws_subnet.public2.id
  route_table_id = aws_route_table.public.id
}


# Private Subnets
resource "aws_subnet" "private1" {
  vpc_id     = aws_vpc.tech_demo_vpc.id
  cidr_block = "10.0.2.0/24"

  availability_zone = "eu-west-2b"
  assign_ipv6_address_on_creation = true
  ipv6_cidr_block = cidrsubnet(aws_vpc.tech_demo_vpc.ipv6_cidr_block, 8, 3)
}

resource "aws_subnet" "private2" {
  vpc_id     = aws_vpc.tech_demo_vpc.id
  cidr_block = "10.0.3.0/24"

  availability_zone = "eu-west-2c"
  assign_ipv6_address_on_creation = true
  ipv6_cidr_block = cidrsubnet(aws_vpc.tech_demo_vpc.ipv6_cidr_block, 8, 4)
}


resource "aws_eip" "nat" {
  domain = "vpc"
}

resource "aws_nat_gateway" "nat_gw" {
  subnet_id     = aws_subnet.public1.id
  allocation_id = aws_eip.nat.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.tech_demo_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gw.id
  }

  route {
    ipv6_cidr_block = "::/0"
    egress_only_gateway_id = aws_egress_only_internet_gateway.eoigw.id
  }
}

resource "aws_route_table_association" "private1" {
  subnet_id      = aws_subnet.private1.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private2" {
  subnet_id      = aws_subnet.private2.id
  route_table_id = aws_route_table.private.id
}

resource "aws_egress_only_internet_gateway" "eoigw" {
  vpc_id = aws_vpc.tech_demo_vpc.id
}


# Security group for webrtc server
# TODO: For prod modify the security of SSH, add VPN and/or whitelist IP's for access to SSH
/*
resource "aws_security_group" "ec2_sg" {
  name        = var.security_group_name
  description = "[NOT FOR PROD] Security group for testing code on EC2 instance, improve safety before production"
  vpc_id      = aws_vpc.tech_demo_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
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

*/
