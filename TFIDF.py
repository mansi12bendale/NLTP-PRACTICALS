# Install these in terminal if not already installed:
# pip install scikit-learn gensim nltk

import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Function to compute TF-IDF
def compute_tfidf(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()
    result = ""
    for i, doc in enumerate(tfidf_array):
        result += f"Document {i+1}:\n"
        for word, score in zip(feature_names, doc):
            if score > 0:
                result += f"  {word}: {score:.4f}\n"
    return result

# Function to compute Word2Vec
def compute_word2vec(corpus):
    tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1, workers=4)
    words = list(model.wv.index_to_key)
    result = "Word Vectors:\n"
    for word in words:
        vector = model.wv[word]
        result += f"{word}: {vector[:5]}...\n"  # Showing first 5 values
    return result

# Button click event
def process_text():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter some text!")
        return
    
    corpus = text.split("\n")  # Each line as a document
    tfidf_result = compute_tfidf(corpus)
    word2vec_result = compute_word2vec(corpus)
    
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "=== TF-IDF Results ===\n")
    output_text.insert(tk.END, tfidf_result)
    output_text.insert(tk.END, "\n=== Word2Vec Results ===\n")
    output_text.insert(tk.END, word2vec_result)

# Create GUI window
root = tk.Tk()
root.title("Text Feature Extractor")

# Input area
tk.Label(root, text="Enter text (one document per line):").pack()
input_text = scrolledtext.ScrolledText(root, width=80, height=10)
input_text.pack()

# Button
tk.Button(root, text="Process Text", command=process_text).pack(pady=5)

# Output area
tk.Label(root, text="Output:").pack()
output_text = scrolledtext.ScrolledText(root, width=80, height=20)
output_text.pack()

# Run the GUI
root.mainloop()
