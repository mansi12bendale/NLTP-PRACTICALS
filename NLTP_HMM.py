import nltk
import math
import pickle
import os
from nltk.corpus import brown
from collections import defaultdict, Counter
import tkinter as tk
from tkinter import scrolledtext

# Ensure necessary corpora are downloaded
nltk.download('brown')
nltk.download('universal_tagset')

# ---------------------------
# Helper function to replace lambda (fix for pickle)
# ---------------------------
def nested_defaultdict():
    return defaultdict(float)

# ---------------------------
# HMM POS TAGGER CLASS
# ---------------------------
class HMMTagger:
    def __init__(self):
        self.transition_probs = defaultdict(nested_defaultdict)
        self.emission_probs = defaultdict(nested_defaultdict)
        self.tag_counts = defaultdict(int)
        self.tags = set()
        self.vocab = set()
        self.trained = False

    def train(self, tagged_sents):
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)

        for sent in tagged_sents:
            prev_tag = "<s>"
            for word, tag in sent:
                word = word.lower()
                self.vocab.add(word)
                emission_counts[tag][word] += 1
                transition_counts[prev_tag][tag] += 1
                self.tag_counts[tag] += 1
                self.tags.add(tag)
                prev_tag = tag
            transition_counts[prev_tag]["</s>"] += 1

        # Compute transition probabilities
        for prev_tag in transition_counts:
            total = sum(transition_counts[prev_tag].values())
            for tag in transition_counts[prev_tag]:
                self.transition_probs[prev_tag][tag] = transition_counts[prev_tag][tag] / total

        # Compute emission probabilities
        for tag in emission_counts:
            total = sum(emission_counts[tag].values())
            for word in emission_counts[tag]:
                self.emission_probs[tag][word] = emission_counts[tag][word] / total

        self.trained = True
        print("✅ Model trained.")

    def viterbi(self, sentence):
        sentence = [word.lower() for word in sentence]
        V = [{}]
        path = {}

        for tag in self.tags:
            trans_p = self.transition_probs["<s>"].get(tag, 1e-6)
            emit_p = self.emission_probs[tag].get(sentence[0], 1e-6)
            V[0][tag] = math.log(trans_p) + math.log(emit_p)
            path[tag] = [tag]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for tag in self.tags:
                (prob, best_prev_tag) = max(
                    (V[t - 1][prev_tag] + math.log(self.transition_probs[prev_tag].get(tag, 1e-6)) +
                     math.log(self.emission_probs[tag].get(sentence[t], 1e-6)), prev_tag)
                    for prev_tag in self.tags
                )
                V[t][tag] = prob
                new_path[tag] = path[best_prev_tag] + [tag]
            path = new_path

        (final_prob, final_tag) = max((V[-1][tag], tag) for tag in self.tags)
        return list(zip(sentence, path[final_tag]))

# ---------------------------
# GUI CLASS
# ---------------------------
class POSApp:
    def __init__(self, root, tagger):
        self.tagger = tagger
        self.root = root
        root.title("HMM POS Tagger")

        self.label = tk.Label(root, text="Enter a sentence:")
        self.label.pack()

        self.text_input = tk.Entry(root, width=100)
        self.text_input.pack(pady=5)

        self.tag_button = tk.Button(root, text="Tag Sentence", command=self.tag_text)
        self.tag_button.pack()

        self.output = scrolledtext.ScrolledText(root, width=100, height=15, wrap=tk.WORD)
        self.output.pack(pady=10)

    def tag_text(self):
        input_text = self.text_input.get()
        words = input_text.strip().split()
        if not words:
            self.output.insert(tk.END, "Please enter a sentence.\n")
            return

        tagged = self.tagger.viterbi(words)
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "Tagged Output:\n\n")
        for word, tag in tagged:
            self.output.insert(tk.END, f"{word}/{tag} ")

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    model_file = "hmm_tagger_model.pkl"
    tagger = HMMTagger()

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            tagger = pickle.load(f)
        print("✅ Loaded trained model.")
    else:
        print("🔁 Training model... (takes a few seconds)")
        tagged_sents = brown.tagged_sents(tagset="universal")
        tagger.train(tagged_sents)
        with open(model_file, "wb") as f:
            pickle.dump(tagger, f)
        print("💾 Model saved.")

    root = tk.Tk()
    app = POSApp(root, tagger)
    root.mainloop()

if __name__ == "__main__":
    main()
