import streamlit as st
import numpy as np
from src.inference import predict_with_attention, tokenizer

st.title("🧠 Attention-based Text Classifier")
user_input = st.text_area("Enter text to classify:", "The movie was absolutely fantastic!")

if st.button("Classify"):
    probs, attentions, inputs = predict_with_attention(user_input)
    label = np.argmax(probs)
    st.write(f"**Prediction:** {'Positive' if label==1 else 'Negative'}")
    st.bar_chart(probs[0])
    # TODO: visualize attention heatmap here
    
