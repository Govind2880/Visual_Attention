import torch
import os
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
import numpy as np

# Try different model paths based on your structure
model_paths = [
    "models",  # Your actual model location
    "./models", 
    "models/best_model",
    "./models/best_model"
]

# Find the correct model path
model_path = None
for path in model_paths:
    model_dir = Path(path)
    if model_dir.exists() and (model_dir / "config.json").exists():
        model_path = str(model_dir)
        print(f"✅ Found model at: {model_path}")
        break

if model_path is None:
    print("⚠️ No fine-tuned model found, using base DistilBERT")
    model_path = "distilbert-base-uncased"

# Global variables for lazy loading
_model = None
_tokenizer = None
_langchain_pipeline = None

def load_model():
    """Lazy load the model and tokenizer."""
    global _model, _tokenizer, _langchain_pipeline
    
    if _model is not None:
        return _model, _tokenizer
    
    try:
        print(f"🔄 Loading model from {model_path}")
        
        # Load model with attention output
        _model = DistilBertForSequenceClassification.from_pretrained(
            model_path, 
            output_attentions=True,
            num_labels=2,  # Binary classification
            local_files_only=True if model_path != "distilbert-base-uncased" else False
        )
        
        # Load tokenizer
        _tokenizer = DistilBertTokenizerFast.from_pretrained(
            model_path,
            local_files_only=True if model_path != "distilbert-base-uncased" else False
        )
        
        print("✅ Model and tokenizer loaded successfully!")
        
        # Try to create LangChain pipeline (optional)
        try:
            from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
            
            hf_pipeline = pipeline(
                "text-classification",
                model=_model,
                tokenizer=_tokenizer,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            _langchain_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)
            print("✅ LangChain pipeline created successfully!")
            
        except ImportError:
            print("⚠️ LangChain not available, skipping pipeline creation")
            _langchain_pipeline = None
        except Exception as e:
            print(f"⚠️ LangChain pipeline creation failed: {e}")
            _langchain_pipeline = None
        
        return _model, _tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        print("🔄 Falling back to base DistilBERT...")
        
        # Fallback to base model
        try:
            _model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2,
                output_attentions=True
            )
            _tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            print("✅ Base DistilBERT loaded as fallback")
            return _model, _tokenizer
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            raise

def predict_with_attention(text):
    """Main prediction function with attention weights."""
    # Ensure model is loaded
    model, tokenizer = load_model()
    
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()
    
    # Get attention weights
    attentions = outputs.attentions  # tuple of attention matrices
    
    return probs, attentions, inputs

def get_langchain_pipeline():
    """Get the LangChain pipeline instance."""
    load_model()  # Ensure model is loaded
    return _langchain_pipeline

# Export tokenizer and model for compatibility
def get_tokenizer():
    """Get the tokenizer instance."""
    _, tokenizer = load_model()
    return tokenizer

def get_model():
    """Get the model instance."""
    model, _ = load_model()
    return model

# For backward compatibility
tokenizer = get_tokenizer()
model = get_model()