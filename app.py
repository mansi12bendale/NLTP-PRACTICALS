import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
def load_data():
    return pd.read_csv("combined_emotion.csv", encoding="latin1")

df = load_data()

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return filtered_tokens, stemmed, lemmatized

# Streamlit UI
st.title("NLP Text Preprocessor with Dataset")
st.write("Enter text below or choose a sample from the dataset.")

# Select text from dataset
if st.checkbox("Use sample from dataset"):
    sample_text = st.selectbox("Select a sample text", df.iloc[:, 0].dropna().unique())
    text_input = sample_text
else:
    text_input = st.text_area("Enter Text:")

if st.button("Process"):
    if text_input:
        tokens, stemmed, lemmatized = preprocess_text(text_input)
        st.subheader("Tokens")
        st.write(tokens)
        st.subheader("Stemmed")
        st.write(stemmed)
        st.subheader("Lemmatized")
        st.write(lemmatized)
    else:
        st.warning("Please enter or select some text to process.")

import nltk
nltk.download('punkt')
