from tabnanny import verbose
import numpy as np
import pickle
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load model
model = load_model("models/lstm_model.h5")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load metadata
with open("models/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

max_sequence_len = metadata["max_sequence_len"]

# def predict_next_word(seed_text, words_to_predict):
#     for _ in range(words_to_predict):
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

#         predicted = np.argmax(model.predict(token_list), axis=-1)

#         for word, index in tokenizer.word_index.items():
#             if index == predicted:
#                 return word
def predict(seed_text, words_to_generate):
    # Repeat the process for as many words as you want
    for _ in range(words_to_generate):
        
        # 1. Convert the text input into numbers (tokens)
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # 2. Make sure the input size matches what the model expects (Padding)
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # 3. Ask the model for the most likely next word (The "Winner")
        predicted = model.predict(token_list, verbose = 0)[0]
        winner_index = np.argmax(predicted) 
        
        # 4. Turn that number back into a word
        next_word = tokenizer.index_word.get(winner_index, "")
        
        # 5. Add that word to our sentence and repeat
        seed_text += " " + next_word
        
    return seed_text
# Example
seed = "hello my name is"
print("Next word:", predict(seed,4))