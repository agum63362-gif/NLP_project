import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("language_dataset.csv", on_bad_lines='skip', engine='python')

# ---------------- DATA CHECK (BEFORE CLEANING) ----------------
print("Before cleaning:")
print(data.isnull().sum())
print("Total rows before cleaning:", len(data))
print()

# Drop missing values
data = data.dropna(subset=["text", "language"])

# Remove empty strings
data = data[(data["text"].str.strip() != "") & (data["language"].str.strip() != "")]

# ---------------- DATA CHECK (AFTER CLEANING) ----------------
print("After cleaning:")
print(data.isnull().sum())
print("Total rows after cleaning:", len(data))
print()

# Split data
X = data["text"]
y = data["language"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

print("Model trained successfully!\n")

# Test in terminal
while True:
    sentence = input("Enter a sentence: ")

    if sentence.lower() == "exit":
        print("Exiting program...")
        break

    sentence_vector = vectorizer.transform([sentence])
    prediction = model.predict(sentence_vector)

    print("Predicted Language:", prediction[0])
    print()