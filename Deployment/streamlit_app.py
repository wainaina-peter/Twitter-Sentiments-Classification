import pickle
import numpy as np
import pandas as pd
import streamlit as st
import string

# Load the model
with open('LRmodel.pkl', 'rb') as model_file:
    classifier_LR = pickle.load(model_file)

# Preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Prediction function
def predict_sentiment(text):
    input_data = np.array([text])
    reshaped_data = input_data.reshape(1, -1)
    prediction = classifier_LR.predict(reshaped_data)
    return prediction[0]

def main():
    st.title('Twitter(X) Sentiments Analyzer')

    # Input variables
    text = st.text_input('Input the tweet here (Paste it)')
    make_prediction = ""

    # Prediction code
    if st.button('Classify Tweet'):
        make_prediction = predict_sentiment(preprocess_text(text))
        sentiment = "Positive" if make_prediction == 1 else "Negative"
        st.success('The tweet is classified as {}'.format(sentiment))

if __name__ == "__main__":
    main()
