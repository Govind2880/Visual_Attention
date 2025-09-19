from rest_framework import serializers

class PredictionRequestSerializer(serializers.Serializer):
    text = serializers.CharField()
