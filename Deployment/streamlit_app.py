import pickle
import numpy as np
import streamlit as st
import os
import string
model = pickle.load(open('LRmodel.pkl', 'rb'))

def main():
    st.title('Twitter(X) Sentiments Analyzer')

    #input variables
    text = st.text_input('Input the tweet here (Paste it)')
    makeprediction = ""


    #prediction code
    if st.button('Classify Tweet'):
        makeprediction = model.predict([text])
        output = round(makeprediction[0],2)
        st.success('The tweet is classified as {}'.format(output))

if __name__=="__main__":
    main()        