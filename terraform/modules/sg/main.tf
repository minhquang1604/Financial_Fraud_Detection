resource "aws_security_group" "this" {
  name        = var.name
  description = var.description
  vpc_id      = var.vpc_id

  tags = merge({ Name = var.name }, var.tags)
}

resource "aws_vpc_security_group_ingress_rule" "this" {
  for_each = { for i, r in var.ingress_rules : i => r }

  security_group_id = aws_security_group.this.id
  cidr_ipv4         = each.value.cidr_ipv4
  from_port         = each.value.protocol == "-1" ? 0 : each.value.from_port
  to_port           = each.value.protocol == "-1" ? 0 : each.value.to_port
  ip_protocol       = each.value.protocol
  description       = each.value.description
}

resource "aws_vpc_security_group_egress_rule" "this" {
  for_each = { for i, r in var.egress_rules : i => r }

  security_group_id = aws_security_group.this.id
  cidr_ipv4         = each.value.cidr_ipv4
  from_port         = each.value.protocol == "-1" ? 0 : each.value.from_port
  to_port           = each.value.protocol == "-1" ? 0 : each.value.to_port
  ip_protocol       = each.value.protocol
  description       = each.value.description
}
