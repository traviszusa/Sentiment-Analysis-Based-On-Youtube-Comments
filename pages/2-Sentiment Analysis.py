import streamlit as st
import pickle
import pandas as pd

# Function to load the model
@st.cache_data
def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict sentiment
def predict_sentiment(model, input_text):
    prediction = model.predict([input_text])
    return prediction[0]

# Main function
def main():
    st.title("Sentiment Analysis Tools")

    model1_filename = './model/naive-bayes-model.pkl'
    model2_filename = './model/naive-bayes-model-balanced.pkl'
    model1 = load_model(model1_filename)
    model2 = load_model(model2_filename)
    
    st.success("Unbalanced and Balanced Models successfully loaded!")
    
    input_text = st.text_input("Enter text to analyze:")
    
    if input_text:
        sentiment_model1 = predict_sentiment(model1, input_text)
        st.info(f"Prediction of Unbalanced Naive Bayes Model: {sentiment_model1}")
        
        sentiment_model2 = predict_sentiment(model2, input_text)
        st.info(f"Prediction of Balanced Naive Bayes Model: {sentiment_model2}")

if __name__ == '__main__':
    main()