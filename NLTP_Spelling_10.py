
import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import pipeline
from spellchecker import SpellChecker

# Initialize tools
spell = SpellChecker()
text_gen = pipeline("text-generation", model="gpt2")

def check_spelling(text):
    words = text.split()
    misspelled = spell.unknown(words)
    return list(misspelled)

def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) if word in spell.unknown([word]) else word for word in words]
    return ' '.join(corrected_words)

def autocomplete_text(prefix, max_length=50):
    completions = text_gen(prefix, max_length=max_length, num_return_sequences=1, do_sample=True)
    return completions[0]['generated_text']

def on_check():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showinfo("Info", "Please enter text.")
        return
    result = check_spelling(text)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Misspelled Words: " + ", ".join(result))

def on_correct():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showinfo("Info", "Please enter text.")
        return
    result = correct_spelling(text)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Corrected Text:\n" + result)

def on_autocomplete():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showinfo("Info", "Please enter text.")
        return
    result = autocomplete_text(text)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Autocomplete Result:\n" + result)

# GUI Layout
root = tk.Tk()
root.title("NLTP Spelling GUI")

tk.Label(root, text="Enter your text:").pack()
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=8)
input_text.pack(padx=10, pady=5)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="Check Spelling", command=on_check).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Correct Spelling", command=on_correct).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Autocomplete", command=on_autocomplete).grid(row=0, column=2, padx=5)

tk.Label(root, text="Output:").pack()
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
output_text.pack(padx=10, pady=5)

root.mainloop()
