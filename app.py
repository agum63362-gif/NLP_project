from flask import Flask, render_template, request
import pandas as pd
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ---------------- LOAD DATA ----------------
def load_dataset(filename):
    texts = []
    languages = []

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                if len(row) > 2:
                    text = ','.join(row[:-1])
                    language = row[-1].strip()
                else:
                    text = row[0].strip()
                    language = row[1].strip()

                if text and language:
                    texts.append(text)
                    languages.append(language)

    return pd.DataFrame({"text": texts, "language": languages})


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# ---------------- TRAIN MODEL ----------------
print("Loading dataset...")
df = load_dataset("language_dataset.csv")

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.strip() != ""]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["language"]

model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully!")


# ---------------- PREDICTION ----------------
def predict_language(text):
    if not text or not text.strip():
        return "Please enter some text"

    text = clean_text(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]


# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    sentence = ""

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        if sentence:
            prediction = predict_language(sentence)

    return render_template("index.html", prediction=prediction, sentence=sentence)

