# train_model.py
import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load data
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df])
df = df[["title", "text", "label"]]
df["content"] = df["title"] + " " + df["text"]
df = df.sample(frac=1).reset_index(drop=True)

# Clean text
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df["content"] = df["content"].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["content"])

X = tokenizer.texts_to_sequences(df["content"])
X = pad_sequences(X, maxlen=200)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=4, batch_size=64, validation_data=(X_test, y_test))

# Save model and tokenizer
model.save("saved_models/fake_news_lstm_model.h5")
with open("saved_models/tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
