
# 📞 Call Sentiment Analysis using Deep Learning

This project implements a sentiment analysis system for analyzing customer call transcripts. It uses both traditional NLP techniques (TextBlob) and deep learning (LSTM-based neural networks) to classify text into **Positive**, **Neutral**, or **Negative** sentiments.

---

## 🔍 Overview

The main objectives of this project are:
- Automatically detect the emotional tone in customer support call transcripts.
- Visualize and report sentiment distribution.
- Train a robust LSTM model for multi-class sentiment classification.
- Predict sentiment of new, unseen texts.

---

## 📁 Project Structure

```

Call-sentiment-analysis-main/
│
├── model.json                      # Model architecture (JSON format)
├── model\_weights.weights.h5       # Trained model weights
├── tokenizer.pickle               # Tokenizer object for inference
├── app.py (optional)              # Streamlit/Flask interface (if needed)
├── your\_dataset.csv               # Input call data file (CSV format)
└── README.md                      # This file

````

---

## 🧠 Technologies Used

- **Python**
- **TensorFlow/Keras** (LSTM, Embedding)
- **TextBlob** (for basic polarity-based sentiment detection)
- **Matplotlib** (for visualization)
- **Scikit-learn** (data processing & train/test split)
- **Pandas & NumPy**

---

## ⚙️ Features

- ✔️ Dataset loading and cleaning
- ✔️ Sentiment labeling using TextBlob (rule-based)
- ✔️ One-hot encoding for training labels
- ✔️ Tokenization + sequence padding
- ✔️ LSTM model with:
  - Embedding layer
  - Spatial dropout
  - Regularization
- ✔️ Early stopping and checkpointing
- ✔️ Model performance visualization
- ✔️ Save and reload model for inference
- ✔️ Predict sentiment of custom text input

---

## 📊 Visualizations

- 📈 **Sentiment distribution (Bar Chart & Pie Chart)**
- 📈 **Training vs Validation Accuracy**
- 📉 **Loss curves** *(optional if added)*
- 📋 **Final classification statistics**

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Call-sentiment-analysis-main.git
cd Call-sentiment-analysis-main
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> You may also need:

```bash
pip install textblob
python -m textblob.download_corpora
```

### 3. Train the Model

Make sure to set the correct path to your CSV dataset:

```python
dataset_path = r"path_to_dataset"
```

Then run the script to:

* Preprocess text
* Generate one-hot encodings
* Train LSTM
* Save model & tokenizer

### 4. Predict on New Data

Update the `text_data` list with any new samples you'd like to test:

```python
text_data = ["This is a great movie!", "I hated the ending."]
```

Then run the prediction block to see the output sentiment.

---

## ✅ Sample Output

```
Text: This is a great movie!
Predicted Sentiment: Positive

Text: I hated the ending of that book.
Predicted Sentiment: Negative
```

---

## 💡 Advantages

* ✅ **Hybrid approach**: Combines rule-based (TextBlob) and deep learning (LSTM) models.
* ✅ **Domain-adaptable**: Can be trained on your own call center data for improved results.
* ✅ **Scalable**: Works with large datasets (uses batch training).
* ✅ **Customizable**: Easily swap the model architecture, tokenizer, or embeddings.
* ✅ **Deployable**: Can be turned into a web app using Streamlit or Flask.

---

## 🧪 Requirements

* Python 3.8+
* TensorFlow/Keras
* TextBlob
* Pandas, NumPy
* Scikit-learn
* Matplotlib

---

## 📂 Tokenizer & Model Saving

* Model saved as: `model.json` + `model_weights.weights.h5`
* Tokenizer saved as: `tokenizer.pickle`
* Can be reloaded easily for deployment or batch inference.
