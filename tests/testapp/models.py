from __future__ import annotations

from django.contrib.auth.models import User
from django.db import models


class Widget(models.Model):
    name = models.CharField(max_length=100)
    price = models.IntegerField(default=0)
    owner = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.CASCADE, related_name="widgets"
    )

    def __str__(self) -> str:
        return self.name
