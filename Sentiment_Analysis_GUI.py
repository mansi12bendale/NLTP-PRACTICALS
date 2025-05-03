import tkinter as tk
from textblob import TextBlob

# Function to perform sentiment analysis
def analyze_sentiment():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        result_label.config(text="Please enter some text.")
        return

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    result_label.config(text=f"Sentiment: {sentiment}\nPolarity Score: {polarity:.2f}")

# GUI setup
root = tk.Tk()
root.title("Sentiment Analysis Tool")
root.geometry("500x350")
root.configure(bg="#f2f2f2")

title_label = tk.Label(root, text="Sentiment Analysis (TextBlob)", font=("Helvetica", 16, "bold"), bg="#f2f2f2")
title_label.pack(pady=10)

entry = tk.Text(root, height=7, width=55, font=("Helvetica", 12))
entry.pack(pady=10)

analyze_button = tk.Button(root, text="Analyze Sentiment", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", command=analyze_sentiment)
analyze_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f2f2f2", fg="#333")
result_label.pack(pady=20)

root.mainloop()
