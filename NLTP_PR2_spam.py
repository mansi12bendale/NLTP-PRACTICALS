import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to load the dataset from the file
def load_dataset():
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        if 'message' not in df.columns or 'label' not in df.columns:
            messagebox.showerror("Error", "Dataset must contain 'message' and 'label' columns.")
            return None
        return df
    else:
        messagebox.showwarning("No File", "No file selected.")
        return None

# Function to train the model
def train_model(df):
    # Preprocess the data
    X = df['message']  # The feature: message content
    y = df['label']    # The target: spam or ham

    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Vectorize the text data (convert text to numerical data)
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    classification_report_text = classification_report(y_test, y_pred)

    return model, vectorizer, accuracy, classification_report_text

# Function to handle file selection and training
def handle_train_button():
    df = load_dataset()
    if df is not None:
        model, vectorizer, accuracy, report = train_model(df)
        accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
        classification_report_label.config(text=report)
        model_data["model"] = model
        model_data["vectorizer"] = vectorizer
        messagebox.showinfo("Training Complete", "Model has been trained successfully!")

# Function to predict a new email
def handle_predict_button():
    new_email = new_email_entry.get()
    if "model" not in model_data:
        messagebox.showerror("Error", "Model is not trained yet. Please train the model first.")
        return

    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    new_email_vec = vectorizer.transform([new_email])
    prediction = model.predict(new_email_vec)
    prediction_label.config(text=f"Prediction: {prediction[0]}")

# Create a Tkinter window
root = tk.Tk()
root.title("Spam Email Detection")

# Create UI elements
train_button = tk.Button(root, text="Train Model", command=handle_train_button)
train_button.pack(pady=10)

accuracy_label = tk.Label(root, text="Accuracy: N/A")
accuracy_label.pack(pady=5)

classification_report_label = tk.Label(root, text="Classification Report: \n(N/A)", justify=tk.LEFT)
classification_report_label.pack(pady=10)

# For prediction
new_email_label = tk.Label(root, text="Enter a new email text:")
new_email_label.pack(pady=5)

new_email_entry = tk.Entry(root, width=50)
new_email_entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=handle_predict_button)
predict_button.pack(pady=10)

prediction_label = tk.Label(root, text="Prediction: N/A")
prediction_label.pack(pady=5)

# Store model and vectorizer in a dictionary
model_data = {}

# Run the GUI application
root.mainloop()
