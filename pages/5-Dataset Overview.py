import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to load the unbalanced dataset
@st.cache_data
def load_data_unbalanced():
    data = pd.read_csv('./data/youtube-comment-cleaned-sentiment.csv')
    return data

# Function to load balanced dataset
@st.cache_data
def load_data_balanced():
    data = pd.read_csv('./data/youtube-comment-cleaned-sentiment-balanced.csv')
    return data

# Main function
@st.cache_data
def main():
    data_unbalanced = load_data_unbalanced()
    data_balanced = load_data_balanced()

    st.subheader("Dataset Unbalanced Overview")
    st.write("A quick look at the dataset unbalanced used by unbalanced model:")
    st.write(data_unbalanced)

    st.subheader("Dataset Balanced Overview")
    st.write("A quick look at the dataset balanced used by balanced model:")
    st.write(data_balanced)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Unbalanced Distribution")
        sentiment_count = data_unbalanced['sentiment_prediction'].value_counts()
        st.bar_chart(sentiment_count, color='#d2a4a7')

        csv_file = './data/youtube-comment-cleaned-sentiment.csv'
        df = pd.read_csv(csv_file)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='gist_heat_r').generate_from_text(' '.join(df['cleaned_stemmed']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    with col2:
        st.subheader("Sentiment Balanced Distribution")
        sentiment_count = data_balanced['sentiment_prediction'].value_counts()
        st.bar_chart(sentiment_count, color='#d2a4a7')

        csv_file = './data/youtube-comment-cleaned-sentiment-balanced.csv'
        df = pd.read_csv(csv_file)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='gist_heat_r').generate_from_text(' '.join(df['cleaned_stemmed']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    st.subheader("Insights Based on the Dataset:")
    st.markdown('''
        1. In this case if the dataset is unbalanced, the sentiment analysis tool will receive more variations of words that can be processed by the model, but the model will be biased towards the sentiment category that has the highest number of sentences.
        2. However, in this case if the dataset is balanced, the sentiment analysis tool will receive less variety of words that can be processed by the model, but the model will be less biased towards one of the sentiment categories because the number is equalized.
    ''')

if __name__ == "__main__":
    main()