import pickle
import numpy as np
import pandas as pd
import streamlit as st
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model
with open('LRmodel.pkl', 'rb') as model_file:
    classifier_LR = pickle.load(model_file)

# Create a WordNetLemmatizer and set of stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Vectorizer for text preprocessing
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)

# Preprocess the text by lemmatizing words and removing stopwords
def preprocess_text(text):
    processed_text = []
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    alpha_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"

    for tweet in text:
        # Convert tweet to lowercase
        tweet = tweet.lower()

        # Replace URLs with 'URL'
        tweet = re.sub(url_pattern, ' URL', tweet)

        # Replace @USERNAME with 'USER'
        tweet = re.sub(user_pattern, ' USER', tweet)

        # Replace non-alphanumeric characters with whitespace
        tweet = re.sub(alpha_pattern, ' ', tweet)

        # Replace three or more consecutive letters with two letters
        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        # Lemmatize words and remove stopwords
        tweet_words = [lemmatizer.lemmatize(word) for word in tweet.split() if word not in stop_words and len(word) > 1]
        processed_text.append(' '.join(tweet_words))

    return processed_text

# Preprocess and vectorize the input text
def preprocess_and_vectorize(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectoriser.transform(preprocessed_text)
    return vectorized_text

# Prediction function
def predict_sentiment(text):
    vectorized_text = preprocess_and_vectorize([text])
    prediction = classifier_LR.predict(vectorized_text)
    return prediction[0]

def main():
    st.title('Twitter(X) Sentiments Analyzer')
    text = st.text_input('Input the tweet here (Paste it)')
    if st.button('Classify Tweet'):
        prediction = predict_sentiment(text)
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success('The tweet is classified as {}'.format(sentiment))

if __name__ == "__main__":
    main()
