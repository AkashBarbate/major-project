import streamlit as st
import torch
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

# === Page Configuration ===
st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Custom Premium CSS ===
st.markdown("""
    <style>
    /* Dark Theme & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Aesthetics */
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 2rem;
    }
    
    /* Glassmorphism Containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    /* Badges for Predicted Categories */
    .styled-badge {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: 0.5s ease-in-out fadeIn;
    }
    
    /* Styling text areas */
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
        color: white !important;
        font-weight: bold;
        border: none !important;
        border-radius: 8px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# === Constants & Thresholds ===
category_names = [
    'Politics', 'Technology', 'Business', 'Economy',
    'International_Affairs', 'Education', 'Health', 'Science/Other'
]
# We map them directly to ensure `transformers-interpret` doesn't complain about lacking label mapping
label2id = {name: i for i, name in enumerate(category_names)}
id2label = {i: name for i, name in enumerate(category_names)}

thresholds = [(0.55432695),(0.5015201),(0.21850389),(0.19308142),(0.19222741),(0.022489678),(0.014320114),(0.23062292)]

@st.cache_resource
def load_models():
    """Loads the tokenizer and the model into memory safely."""
    tokenizer_path = "models/final_optimized_model/final_tokenizer"
    model_path = "models/final_optimized_model/final_model"
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
    # Force apply id mapping so Explainer works flawlessly
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    model.eval()
    return tokenizer, model

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\\S+|www\\.\\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\\s.,!?;:-]', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# === Application Layout ===
st.markdown("<h1>📰 AI News Classifier & Logic Visualizer</h1>", unsafe_allow_html=True)

# 1. Inputs
with st.container():
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.subheader("Input Article Here")
    title_val = st.text_input("Headline / Title", placeholder="e.g. OpenAI announces new breakthroughs...")
    content_val = st.text_area("Article Content", height=200, placeholder="Paste the full article text here...")
    analyze_btn = st.button("🚀 Analyze & Extract Insights")
    st.markdown('</div>', unsafe_allow_html=True)

# 2. Logic & Outputs
if analyze_btn and content_val:
    with st.spinner("Analyzing text with RoBERTa..."):
        # Predict
        tokenizer, model = load_models()
        device = torch.device('cpu') # Quantized runs beautifully on cpu
        
        cleaned_title = clean_text(title_val)
        cleaned_text = clean_text(content_val)
        combined_text = cleaned_title + ' - ' + cleaned_text
        
        encoding = tokenizer(
            combined_text, add_special_tokens=True, max_length=256,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask']).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Threshold logic
        predicted_labels = []
        scores = []
        for i in range(len(category_names)):
            if probs[i] > thresholds[i]:
                predicted_labels.append(category_names[i])
                scores.append(probs[i])
                
        # Reveal Predictions
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Classification Results")
        
        if not predicted_labels:
            st.warning("No categories strongly aligned with this article.")
        else:
            # Badges
            badges_html = " ".join([f'<span class="styled-badge">{label}</span>' for label in predicted_labels])
            st.markdown(badges_html, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # --- Explainable AI Section ---
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.subheader("🧠 Explainable AI: Keyword Impact")
            st.markdown("Discover the specific underlying words that pushed the model to predict these categories.")
            
            explainer = SequenceClassificationExplainer(model, tokenizer)
            
            tabs = st.tabs(predicted_labels)
            
            for idx, label in enumerate(predicted_labels):
                with tabs[idx]:
                    with st.spinner(f"Generating XAI metrics for {label}..."):
                        # Get attributions
                        word_attr = explainer(combined_text, class_name=label)
                        word_dict = dict(word_attr)
                        
                        # Filter & Sort
                        filtered_scores = {k: v for k, v in word_dict.items() if k not in ['<s>', '</s>', '<pad>', '#s', '#/s']}
                        
                        # Prepare dataframe for charting the top 15 words by absolute impact
                        sorted_items = sorted(filtered_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
                        
                        if sorted_items:
                            df = pd.DataFrame(sorted_items, columns=['Keyword', 'Impact Weight'])
                            df.set_index('Keyword', inplace=True)
                            
                            # We can draw it! Positive vs Negative colors
                            st.bar_chart(df)
                            
                            with st.expander("Show raw Dictionary output"):
                                st.json({k: round(v, 4) for k, v in sorted_items})
                        else:
                            st.info("Not enough words to generate impact metrics.")
            
            st.markdown('</div>', unsafe_allow_html=True)