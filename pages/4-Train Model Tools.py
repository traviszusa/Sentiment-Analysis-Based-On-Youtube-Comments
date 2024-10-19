import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import pickle
import os

# Function to train and save the model
def train_model(data, model_filename, algorithm):
    # Check if the necessary columns exist in the dataframe
    if 'cleaned_stemmed' not in data.columns or 'sentiment_prediction' not in data.columns:
        st.warning("The CSV file must have 'cleaned_stemmed' and 'sentiment_prediction' columns.")
        return None

    X = data['cleaned_stemmed']
    y = data['sentiment_prediction']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select algorithm based on user input
    if algorithm == "Naive Bayes":
        model = make_pipeline(CountVectorizer(), MultinomialNB())
    elif algorithm == "Logistic Regression":
        model = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
    elif algorithm == "Support Vector Classifier":
        model = make_pipeline(CountVectorizer(), SVC())
    else:
        st.warning("Unknown algorithm.")
        return None

    # Train the model
    model.fit(X_train, y_train)

    # Save the model to a file
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    st.success(f"Model successfully trained with {algorithm} and saved to {model_filename}")
    return model_filename

# Function to provide file download
def download_file(file_path):
    with open(file_path, "rb") as file:
        btn = st.download_button(
            label="Download model",
            data=file,
            file_name=os.path.basename(file_path),
            mime="application/octet-stream"
        )
    return btn

# Streamlit UI
st.title("Train Sentiment Model from CSV")

st.markdown('''
    How to use this tool:
    1. Upload a CSV file 
    2. Enter a filename to save the model
    3. Select an algorithm to train the model
    4. Click the 'Train Model' button
    5. Wait for the model to be trained
    6. Download the trained model
''')

st.warning("Do not switch to another tab on the sidebar until the model training is completed!")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Upload a CSV file with 'cleaned_stemmed' column for text and 'sentiment_prediction' column for sentiment labels", type=["csv"])

# Input for model filename
model_filename = st.text_input("Enter the file name to save the model (end with .pkl)", value="sentiment_model.pkl")
model_filename = model_filename if model_filename.endswith(".pkl") else model_filename + ".pkl"

# Dropdown for selecting algorithm
algorithm = st.selectbox(
    "Select an algorithm to train the model",
    ("Naive Bayes", "Logistic Regression", "Support Vector Classifier")
)

# Train model button
if uploaded_file is not None and st.button("Train Model"):
    try:
        # Read the CSV file directly without saving it as a temporary file
        data = pd.read_csv(uploaded_file)

        # Train the model and save it to a file
        model_file = train_model(data, model_filename, algorithm)

        # If the model was trained, provide an option to download it
        if model_file:
            st.subheader("Download Trained Model")
            download_file(model_file)

    except Exception as e:
        st.error(f"Error Occurred: {str(e)}")