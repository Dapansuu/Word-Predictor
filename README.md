# 🧠 Next Word Predictor using LSTM

A deep learning-based **Next Word Prediction model** built using Long Short-Term Memory (LSTM) networks.  
The model learns sequential language patterns from a text corpus and predicts the most probable next word given an input sequence.

---

## 📌 Project Overview

This project implements a word-level language model trained to predict the next word in a sentence.

It demonstrates:

- Text preprocessing and tokenization  
- Sequence generation for supervised learning  
- LSTM-based sequence modeling  
- Autoregressive text generation  
- Model saving and standalone inference  

The model is built using Keras from TensorFlow.

---

## 🏗 Model Architecture

Embedding Layer
↓
LSTM Layer
↓
Dense Layer (Softmax Output)


### Layer Details

- **Embedding Layer**  
  Converts word indices into dense vector representations.

- **LSTM Layer**  
  Learns contextual and sequential dependencies in text.

- **Dense + Softmax Layer**  
  Outputs a probability distribution across the vocabulary.

---

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Next-Word-Predictor.git
cd Next-Word-Predictor
```
### Install required dependencies:
```bash
pip install tensorflow numpy
```

## 🚀 Training the Model
run LSTM.ipynb

## 🔮 Generating Text
```bash
python predict.py
```

## 🧠 How It Works

- Text is tokenized into integer sequences.
- Input sequences are padded to a fixed length.
- The LSTM is trained to predict the next word.

### During Inference

- The model predicts the most probable next word.
- The predicted word is appended to the sentence.
- The process repeats for **n** words (autoregressive generation).

---

## 📊 Dataset

The model can be trained on:

- Custom text corpus  
- Classic literature  
- Wikipedia-based datasets  
- Any clean `.txt` file  

For best performance, use a larger structured dataset.

