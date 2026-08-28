from __future__ import annotations

from rest_framework import serializers
from rest_framework_services import MARKING, FieldMarking

from tests.testapp.models import Widget


class WidgetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Widget
        fields = ["id", "name", "price"]


class WidgetInputSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    price = serializers.IntegerField(min_value=0)


class AgentWidgetSerializer(serializers.ModelSerializer):
    """The widget serializer with agent markings, for audience-projection tests."""

    status = serializers.ChoiceField(
        choices=[("IN_STOCK", "In stock"), ("BACKORDER", "On backorder")],
        default="IN_STOCK",
    )

    class Meta:
        model = Widget
        fields = ["id", "name", "price", "status"]
        extra_kwargs = {
            "id": {"style": {MARKING: FieldMarking.handle("Widget handle.")}},
            "name": {"style": {MARKING: FieldMarking.label()}},
            "price": {"style": {MARKING: FieldMarking.hidden()}},
        }
