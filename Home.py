import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def main():
    # Set up the title and description for the app
    st.title("YouTube Comment Sentiment Analysis")
    st.markdown("""
            <style>
            .video-container {
                text-align: center;  /* Center the video */
                margin: 20px 0;  /* Add some margin for spacing */
            }
            .video-container iframe {
                width: 80%;  /* Adjust the width of the video */
                max-width: 1000px;  /* Set max width to avoid being too large */
                height: 300px;  /* Set the height */
                border-radius: 10px;  /* Rounded corners */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Add a subtle shadow */
            }
            </style>
            <div class="video-container">
                <iframe src="https://www.youtube.com/embed/oG852gUrDG8?si=Rt-uOrdUgcVcH6bi" frameborder="0" allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
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

if __name__ == "__main__":
    main()