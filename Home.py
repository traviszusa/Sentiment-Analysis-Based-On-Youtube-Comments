import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set up the title and description for the app
st.title("YouTube Comment Sentiment Analysis")
st.markdown("""
    Welcome to the YouTube Comment Sentiment Analysis tool! This application classifies comments into three categories:
    - **Positive**
    - **Neutral**
    - **Negative**

    Explore the sentiment distribution of the dataset and use the machine learning model to predict sentiments of new comments.
""")
st.markdown("""
    **How to Use**: Navigate to the sidebar to explore the sentiment prediction tool and more features.
""")

@st.cache_data
def load_data_unbalanced():
    # Example of loading a sample dataframe (adjust to load your dataset)
    data = pd.read_csv('./data/youtube-comment-cleaned-sentiment.csv')
    return data

@st.cache_data
def load_data_balanced():
    data = pd.read_csv('./data/youtube-comment-cleaned-sentiment-balanced.csv')
    return data

data_unbalanced = load_data_unbalanced()
data_balanced = load_data_balanced()

# Display data unbalanced overview
st.subheader("Dataset Unbalanced Overview")
st.write("A quick look at the dataset unbalanced:")
st.write(data_unbalanced)

st.subheader("Dataset Balanced Overview")
st.write("A quick look at the dataset balanced:")
st.write(data_balanced)

col1, col2 = st.columns(2)

with col1:
    # Display sentiment unbalanced distribution
    st.subheader("Sentiment Unbalanced Distribution")
    sentiment_count = data_unbalanced['sentiment_prediction'].value_counts()
    st.bar_chart(sentiment_count)

    # Visualisasi wordcloud dari csv file
    csv_file = './data/youtube-comment-cleaned-sentiment.csv'
    df = pd.read_csv(csv_file)
    wordcloud = WordCloud(width=800, height=400).generate_from_text(' '.join(df['cleaned_stemmed']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

with col2:
    # Display sentiment Balanced distribution
    st.subheader("Sentiment Balanced Distribution")
    sentiment_count = data_balanced['sentiment_prediction'].value_counts()
    st.bar_chart(sentiment_count)

    # Visualisasi wordcloud dari csv file
    csv_file = './data/youtube-comment-cleaned-sentiment-balanced.csv'
    df = pd.read_csv(csv_file)
    wordcloud = WordCloud(width=800, height=400).generate_from_text(' '.join(df['cleaned_stemmed']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)