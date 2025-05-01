import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reversed_word_index = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.keras')

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, "?") for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in  words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

example_revew = "The movie was bad.Do not watch the movie. It wasted time. You have to watch again.Bad bad bad. You should try something else. You should not watch. It is very bad.  Watch something else. This is a failed movie. Do not watch the movie. Acting is not good. Actore are just time passing. You have to just waste your time."
predict_sentiment(example_revew)

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = "Postive" if prediction[0][0] >0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st

st.title("IMDB movie reviewe sentiment analysis")
st.write("Enter a moview review to classify it as positive or negative")
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = "Postive" if prediction[0][0] >0.5 else 'Negative'
    
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")
else:
    st.write("Please enter a movie review")