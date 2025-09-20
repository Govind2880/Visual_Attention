import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# optionally disable wandb prompts
os.environ["WANDB_DISABLED"] = "true"

import gc
import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm.auto import tqdm

# ---------------------
# 1) Prepare dataset
# ---------------------
dataset = load_dataset("imdb")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    # reduce max_length to save memory
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize_function, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Use subset to keep iterations short during debugging; remove selection for full run
train_dataset = tokenized["train"].shuffle(seed=42).select(range(10000))
eval_dataset = tokenized["test"].shuffle(seed=42).select(range(2000))

# ---------------------
# 2) Model (small & optimized)
# ---------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    output_attentions=True
)

# enable memory saving strategies
model.gradient_checkpointing_enable()
model.config.use_cache = False  # avoid caching during training (saves memory)

# ---------------------
# 3) TrainingArguments - NO eval during training
# ---------------------
training_args = TrainingArguments(
    output_dir="./results",
    # disable evaluation during training to avoid eval OOM
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,    # small to reduce peak memory
    gradient_accumulation_steps=4,    # effective batch = 16 (4 * 4)
    per_device_eval_batch_size=4,     # used only if you manually call trainer.evaluate() on GPU
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,                         # mixed precision if GPU supports it
    report_to="none",
    ddp_find_unused_parameters=False,
)

# ---------------------
# 4) Metrics function (used only for CPU eval later)
# ---------------------
def compute_metrics_from_preds(labels, preds):
    preds = np.asarray(preds)
    preds_cls = preds.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_cls, average="binary")
    acc = accuracy_score(labels, preds_cls)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ---------------------
# 5) Trainer setup (train only)
# ---------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# ---------------------
# 6) Free caches and run training on GPU
# ---------------------
gc.collect()
torch.cuda.empty_cache()

trainer.train()

# save the trained model (GPU-trained)
trainer.save_model("./models/best_model")
tokenizer.save_pretrained("./models/best_model")

print("✅ Training done and model saved to ./models/best_model")

# ---------------------
# 7) CPU Evaluation (avoid GPU OOM) - iterative batch inference on CPU
# ---------------------
print("➡ Moving model to CPU for evaluation to avoid GPU OOM...")
device_cpu = torch.device("cpu")
model.to(device_cpu)
model.eval()

# dataloader on CPU
eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc="CPU evaluation"):
        input_ids = batch["input_ids"].to(device_cpu)
        attention_mask = batch["attention_mask"].to(device_cpu)
        labels = batch["labels"].to(device_cpu)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.detach().cpu().numpy()
        all_preds.append(logits)
        all_labels.append(labels.detach().cpu().numpy())

# concat results
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

metrics = compute_metrics_from_preds(all_labels, all_preds)
print("✅ Evaluation metrics (computed on CPU):", metrics)

# ---------------------
# 8) cleanup
# ---------------------
del model
gc.collect()
torch.cuda.empty_cache()
print("Done.")
