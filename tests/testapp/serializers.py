from __future__ import annotations

from rest_framework import serializers

from tests.testapp.models import Widget


class WidgetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Widget
        fields = ["id", "name", "price"]


class WidgetInputSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    price = serializers.IntegerField(min_value=0)
