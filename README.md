🧠 RNN Sentiment Analysis with TensorFlow

This project builds a Recurrent Neural Network (RNN) using TensorFlow/Keras to analyze customer sentiment from clothing reviews. It classifies reviews as positive (recommended) or negative (not recommended).

📦 Dataset Overview

👕 Clothing review dataset

📊 23,472 rows and 10 columns

🎯 Target: Sentiment based on Rating (binary: 1 if rating > 3, else 0)

🛠️ Libraries Used

Data & Visualization:

pandas, numpy

matplotlib, seaborn, plotly

NLP:

nltk (lemmatization, stopwords, tokenization)

re (text cleaning)

ML/DL:

tensorflow, keras

scikit-learn

🧾 Key Concepts

🔁 RNN: Handles sequential data using hidden states

🧼 Text Preprocessing:

Lowercase conversion

Removing stopwords

Lemmatization

Removing punctuation

🔤 Tokenization: Converts text to numerical sequences

🧱 Padding: Makes all sequences the same length

🧠 Model: Embedding → RNN → Dense → Output

🧪 Model Summary
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=40),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])


✅ Loss: Binary Crossentropy

⚙️ Optimizer: Adam

📏 Metric: Accuracy

🔁 Epochs: 5

🔡 Input Length: 40

📖 Vocabulary Size: 10,000

📈 EDA & Visualizations

📦 Count plots for clothing classes & ratings

📊 Age distribution by sentiment & rating

📤 Histograms with boxplots for deeper insight

🚀 Steps to Run

📥 Install dependencies:

pip install pandas numpy matplotlib seaborn plotly scikit-learn nltk tensorflow


🔍 Download NLTK data:

import nltk
nltk.download('all')


▶️ Run the script or notebook:

python rnn_sentiment_analysis.py

✅ Results

Model trained for 5 epochs

Successfully predicts positive vs. negative sentiment

Handles text input efficiently using RNN layers

📌 Future Improvements

🔁 Try LSTM or GRU instead of SimpleRNN

🔍 Hyperparameter tuning

🧠 Use pretrained embeddings like GloVe

📉 Add validation data for better generalization
