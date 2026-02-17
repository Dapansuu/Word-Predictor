import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore

# Load model and tokenizer
model = tf.keras.models.load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

MAX_SEQUENCE_LENGTH = metadata["max_sequence_len"]


st.title("LSTM Next Word Predictor")

text_input = st.text_input("Enter text:")

if st.button("Predict"):
    sequence = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

    prediction = model.predict(padded)
    predicted_index = np.argmax(prediction)

    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    predicted_word = reverse_word_index.get(predicted_index, "")

    st.success(f"Next word: {predicted_word}")
