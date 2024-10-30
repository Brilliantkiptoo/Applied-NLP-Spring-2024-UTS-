import streamlit as st
import pandas as pd
from model1_doc2vec import run_doc2vec
from model2_lstm import run_lstm
from model3_bert import run_bert

st.title("Financial News Sentiment Analysis App")

# Upload a CSV file for Doc2Vec and LSTM models
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

# Text input for BERT model
user_input = st.text_area("Enter news text for BERT analysis:")

# Model 1: Doc2Vec
if st.button("Run Doc2Vec"):
    if uploaded_file is not None:
        predictions, y_test = run_doc2vec(df, text_column='news_text', target_column='sentiment')
        st.write(f"Doc2Vec Predictions: {predictions}")

# Model 2: LSTM
if st.button("Run LSTM"):
    if uploaded_file is not None:
        predictions, y_test = run_lstm(df, text_column='news_text', target_column='sentiment')
        st.write(f"LSTM Predictions: {predictions}")

# Model 3: BERT
if st.button("Run BERT"):
    prediction = run_bert(user_input)
    st.write(f"BERT Prediction: {prediction}")
