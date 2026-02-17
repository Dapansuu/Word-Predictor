import numpy as np
import pickle
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
import os

# ---------------------------
# Load Model + Tokenizer
# ---------------------------
model = tf.keras.models.load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

MAX_SEQUENCE_LENGTH = metadata["max_sequence_len"]  # Same length used during training

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="LSTM Text Prediction API")

# Request Body Schema
class TextRequest(BaseModel):
    text: str

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict(request: TextRequest):

    # Tokenize input
    sequence = tokenizer.texts_to_sequences([request.text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

    # Predict
    prediction = model.predict(padded)
    predicted_index = np.argmax(prediction)

    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return {"input": request.text, "prediction": word}

    return {"prediction": "Word not found"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
