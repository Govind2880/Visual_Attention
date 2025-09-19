import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model_path = "./models/best_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

def predict_with_attention(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()
    attentions = outputs.attentions
    return probs, attentions, inputs, tokenizer
