import streamlit as st
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Fungsi untuk memuat model dari file .pkl
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
    st.title("Sentiment Analysis Based On Youtube Comments")
    st.write("Keras! Rocky Singgung Penjilat, Silfester Emosi hingga Keluar Kata Kasar - Rakyat Bersuara 03/09")
    st.write("Link: https://youtu.be/oG852gUrDG8?si=iBaeb2cYpCcCSvE8")

    # Muat kedua model dari file .pkl
    model1_filename = './model/naive-bayes-model.pkl'
    model2_filename = './model/naive-bayes-model-balanced.pkl'
    model1 = load_model(model1_filename)
    model2 = load_model(model2_filename)
    
    st.success("Model 1 dan Model 2 berhasil dimuat!")
    
    # Input teks untuk analisis sentimen
    input_text = st.text_input("Masukkan teks untuk dianalisis:")
    
    if input_text:
        sentiment_model1 = predict_sentiment(model1, input_text)
        st.info(f"Prediksi Model Naive Bayes Unbalanced: {sentiment_model1}")
        
        sentiment_model2 = predict_sentiment(model2, input_text)
        st.info(f"Prediksi Model Naive Bayes Balanced: {sentiment_model2}")

    # Visualisasi wordcloud dari csv file
    csv_file = './data/youtube-comment-sentiment-cleaned.csv'
    df = pd.read_csv(csv_file)
    wordcloud = WordCloud(width=800, height=400).generate_from_text(' '.join(df['cleaned_stemmed']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.pyplot(plt)

if __name__ == '__main__':
    main()