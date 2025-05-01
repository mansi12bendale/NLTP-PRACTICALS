from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict
import random
import re

app = Flask(__name__)
CORS(app)

model = {}
n = 2

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    return tokens

def build_ngram_model(tokens, n):
    temp_model = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        temp_model[context][next_word] += 1
    return temp_model

def convert_to_probabilities(temp_model):
    for context in temp_model:
        total = float(sum(temp_model[context].values()))
        for word in temp_model[context]:
            temp_model[context][word] /= total
    return temp_model

def generate_text(model, n, num_words, seed=None):
    if not model:
        return "No model trained yet."
    if seed:
        seed = tuple(seed.lower().split()[-(n-1):])
    else:
        seed = random.choice(list(model.keys()))
    output = list(seed)
    for _ in range(num_words):
        context = tuple(output[-(n-1):])
        if context not in model:
            break
        next_words = list(model[context].keys())
        probabilities = list(model[context].values())
        next_word = random.choices(next_words, probabilities)[0]
        output.append(next_word)
    return ' '.join(output)

@app.route('/train', methods=['POST'])
def train():
    global model, n
    data = request.json
    text = data.get("text", "")
    n = int(data.get("n", 2))
    tokens = preprocess(text)
    temp_model = build_ngram_model(tokens, n)
    model = convert_to_probabilities(temp_model)
    return jsonify({"message": f"{n}-gram model trained successfully", "tokens": len(tokens)})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    num_words = int(data.get("num_words", 20))
    seed = data.get("seed", None)
    generated = generate_text(model, n, num_words, seed)
    return jsonify({"generated_text": generated})

if __name__ == '__main__':
    app.run(debug=True)
