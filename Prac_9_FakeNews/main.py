# main.py
import tkinter as tk
from tkinter import messagebox
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load model and tokenizer
model = load_model("saved_models/fake_news_lstm_model.h5")
with open("saved_models/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def predict_news(news_text):
    cleaned = clean_text(news_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]
    return "Real News ✅" if pred > 0.5 else "Fake News ❌"

def on_predict():
    news = text_input.get("1.0", tk.END)
    if news.strip() == "":
        messagebox.showwarning("Input Error", "Please enter news text.")
        return
    result = predict_news(news)
    messagebox.showinfo("Prediction Result", result)

# GUI Setup
window = tk.Tk()
window.title("Fake News Classifier")
window.geometry("500x400")

tk.Label(window, text="Enter News Text:", font=("Arial", 14)).pack(pady=10)
text_input = tk.Text(window, height=10, wrap=tk.WORD)
text_input.pack(padx=10, pady=10)

tk.Button(window, text="Check News", command=on_predict, bg="blue", fg="white", font=("Arial", 12)).pack(pady=20)

window.mainloop()
