import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import pipeline
import re
import ast
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Pre-trained sentiment analysis model
distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True,
    truncation=True
)

# Function to clean text
def clean_comment(comment, replace_word):
    if pd.isna(comment):
        return ' '
    comment = str(comment)
    comment = re.sub(r'\n+', ' ', comment)
    comment = re.sub(r'http\S+|www\S+', ' ', comment, flags=re.MULTILINE)
    comment = re.sub(r'[^\w\s,]', ' ', comment)
    comment = re.sub(r'[^A-Za-z\s]', ' ', comment)
    comment = comment.lower()
    
    # Apply user-defined replace word dictionary
    for word, replace in replace_word.items():
        comment = re.sub(r'\b' + re.escape(word) + r'\b', replace, comment, flags=re.IGNORECASE)
    
    comment = re.sub(r'\b[a-zA-Z]\b', ' ', comment)
    comment = re.sub(r'(.)\1{2,}', r'\1\1', comment)
    comment = re.sub(r'\s+', ' ', comment).strip()
    
    return comment

# Function to tokenize text
def tokenize_comment(comment):
    return word_tokenize(comment)

# Function to remove stopwords
def stopwords_comment(tokens, additional_stopwords):
    cleaned_token = []
    all_stopwords = set(stopwords.words('english')).union(set(stopwords.words('indonesian'))).union(set(additional_stopwords))
    for token in tokens:
        if token not in all_stopwords:
            cleaned_token.append(token)
    return cleaned_token

# Stemmer setup
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# Function to stem text
def stemming_comment(tokens):
    return [stemmer.stem(token) for token in tokens]

# Function to clean and convert tokenized/stemmed text to a single string
def clean_and_convert(text):
    if isinstance(text, list):
        return ' '.join(text)
    elif isinstance(text, str):
        try:
            word_list = ast.literal_eval(text)
            if isinstance(word_list, list):
                return ' '.join(word_list)
            else:
                cleaned_text = re.sub(r"[\'\[\],]", "", text)
                return cleaned_text.strip()
        except (ValueError, SyntaxError):
            cleaned_text = re.sub(r"[\'\[\],]", "", text)
            return cleaned_text.strip()
    else:
        return ''

# Function to perform sentiment analysis
def sentiment_analysis(text):
    result = distilled_student_sentiment_classifier(text)
    highest_score_label = max(result[0], key=lambda x: x['score'])['label']
    return highest_score_label

# Initialize session state for additional stopwords and uploaded file
if 'additional_stopwords' not in st.session_state:
    st.session_state['additional_stopwords'] = []

if 'replace_word' not in st.session_state:
    st.session_state['replace_word'] = {}

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# Streamlit UI
st.title("Sentiment Analysis with Custom Stopwords and Replace Words")

# Sidebar: Input for Replace Word
st.sidebar.subheader("Add Custom Replace Words")
replace_key = st.sidebar.text_input("Word to Replace")
replace_value = st.sidebar.text_input("Replacement Word")

# Button to add the replace word pair to the dictionary
if st.sidebar.button("Add Replace Word"):
    if replace_key and replace_value:
        st.session_state['replace_word'][replace_key] = replace_value
    else:
        st.sidebar.warning("Both 'Word to Replace' and 'Replacement Word' must be provided")

# Display current replace word dictionary
st.sidebar.subheader("Current Replace Words")
st.sidebar.write(st.session_state['replace_word'])

# Sidebar: Input for additional stopwords
st.sidebar.subheader("Add Custom Stopwords")
additional_stopwords_input = st.sidebar.text_area(
    "Enter additional stopwords separated by commas", 
    value=", ".join(st.session_state['additional_stopwords'])  # Maintain state here
)

# Update session state with additional stopwords
st.session_state['additional_stopwords'] = [word.strip() for word in additional_stopwords_input.split(',')] if additional_stopwords_input else []

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file with a 'comment' column", type=["csv"])

# Maintain the file upload in session state
if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

# If there is a file in session state, process it
if st.session_state['uploaded_file']:
    comment_df = pd.read_csv(st.session_state['uploaded_file'])

    st.subheader("Uploaded Data")
    st.write(comment_df)

    if 'comment' in comment_df.columns:
        st.subheader("Processing Data...")

        total_comments = len(comment_df)

        # Processing the data in steps with progress updates
        comment_df['cleaned_comment'] = comment_df['comment'].apply(lambda x: clean_comment(x, st.session_state['replace_word']))
        comment_df['tokenized_comment'] = comment_df['cleaned_comment'].apply(tokenize_comment)
        comment_df['stopwords_comment'] = comment_df['tokenized_comment'].apply(lambda tokens: stopwords_comment(tokens, st.session_state['additional_stopwords']))
        comment_df['stemmed_comment'] = comment_df['stopwords_comment'].apply(stemming_comment)
        comment_df['cleaned_stemmed'] = comment_df['stemmed_comment'].apply(clean_and_convert)
        comment_df = comment_df.drop_duplicates(subset='cleaned_stemmed')

        st.subheader("Performing Sentiment Analysis...")

        # Initialize progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Sentiment analysis with progress
        sentiments = []
        for idx, comment in enumerate(tqdm(comment_df['cleaned_stemmed']), start=1):
            sentiment = sentiment_analysis(comment)
            sentiments.append(sentiment)

            # Update progress bar and text
            progress_percentage = int((idx / total_comments) * 100)
            progress_bar.progress(progress_percentage)
            progress_text.text(f"Processing: {idx} of {total_comments} ({progress_percentage}%)")

        comment_df['sentiment_prediction'] = sentiments

        st.subheader("Processed Data with Sentiment")
        st.write(comment_df)

        csv = comment_df.to_csv(index=False)
        st.download_button(label="Download Processed CSV", data=csv, file_name='processed_sentiment.csv', mime='text/csv')

        # Reset progress bar
        progress_bar.progress(100)
        progress_text.text("Processing completed!")
    else:
        st.error("The uploaded file does not contain a 'comment' column.")