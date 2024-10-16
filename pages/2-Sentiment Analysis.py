import streamlit as st
import pickle
import pandas as pd

# Fungsi untuk memuat model dari file .pkl
@st.cache_data
def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk prediksi sentimen dari input pengguna
def predict_sentiment(model, input_text):
    prediction = model.predict([input_text])
    return prediction[0]

# Fungsi untuk menampilkan aplikasi Streamlit
def main():
    st.title("Sentiment Analysis Tool")

    # Muat kedua model dari file .pkl
    model1_filename = './model/naive-bayes-model.pkl'
    model2_filename = './model/naive-bayes-model-balanced.pkl'
    model1 = load_model(model1_filename)
    model2 = load_model(model2_filename)
    
    st.success("Model Unbalanced dan Model Balanced berhasil dimuat!")
    
    # Input teks untuk analisis sentimen
    input_text = st.text_input("Masukkan teks untuk dianalisis:")
    
    if input_text:
        sentiment_model1 = predict_sentiment(model1, input_text)
        st.info(f"Prediksi Model Naive Bayes Unbalanced: {sentiment_model1}")
        
        sentiment_model2 = predict_sentiment(model2, input_text)
        st.info(f"Prediksi Model Naive Bayes Balanced: {sentiment_model2}")

# Memulai aplikasi
if __name__ == '__main__':
    main()