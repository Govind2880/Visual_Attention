# train_classifier.py
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# 1. Load Dataset
dataset = load_dataset("imdb")  # 25k train, 25k test

# 2. Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Format for PyTorch
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))  # subset to 10k for faster training
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))

# 4. Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, output_attentions=True
)

# 5. Training Setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
)

# 6. Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average="binary")
    acc = accuracy_score(pred.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train
trainer.train()

# 9. Evaluate
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)

# 10. Save Model
model.save_pretrained("./models/best_model")
tokenizer.save_pretrained("./models/best_model")

print("✅ Model and tokenizer saved to ./models/best_model")
