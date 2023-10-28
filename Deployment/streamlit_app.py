import pickle
import numpy as np
import streamlit as st
import os
import string

model = pickle.load(open('LRmodel.pkl', 'rb'))

def main():
    st.title('Twitter(X) Sentiments Analyzer')

    # Input variables
    text = st.text_input('Input the tweet here (Paste it)')
    makeprediction = ""

    # Prediction code
    if st.button('Classify Tweet'):
        input_data = np.array([text])
        reshaped_data = input_data.reshape(1, -1)
        makeprediction = model.predict(reshaped_data)
        output = round(makeprediction[0], 2)
        st.success('The tweet is classified as {}'.format(output))

if __name__ == "__main__":
    main()
