from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .inference import predict_with_attention
from src.visualize import show_attention
from .serializers import PredictionRequestSerializer

CLASSES = ["Negative", "Positive"]

def home(request):
    if request.method == "POST":
        text = request.POST.get("user_input")
        probs, attentions, inputs, tokenizer = predict_with_attention(text)
        label_idx = probs.argmax()
        attention_html = show_attention(text, tokenizer, attentions, inputs)

        return render(request, "index.html", {
            "text": text,
            "prediction": CLASSES[label_idx],
            "probabilities": zip(CLASSES, probs[0]),
            "attention_html": attention_html
        })

    return render(request, "index.html")

@api_view(["POST"])
def predict_api(request):
    serializer = PredictionRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    text = serializer.validated_data["text"]

    probs, attentions, inputs, tokenizer = predict_with_attention(text)
    label_idx = probs.argmax()
    attention_html = show_attention(text, tokenizer, attentions, inputs)

    return Response({
        "text": text,
        "prediction": CLASSES[label_idx],
        "probabilities": probs[0].tolist(),
        "attention_html": attention_html
    })
