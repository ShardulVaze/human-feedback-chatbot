import pandas as pd
import os
import torch
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

FILE = "feedback_data.csv"

# Global vectorizer (used in training)
vectorizer = TfidfVectorizer(max_features=500)


def save_feedback(prompt, response, label):
    file_exists = os.path.exists(FILE)

    with open(FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["prompt", "response", "label"])

        writer.writerow([prompt, response, label])


def load_data():
    if not os.path.exists(FILE):
        return None, None

    df = pd.read_csv(FILE)

    df = df.dropna()
    df = df[["prompt", "response", "label"]]

    texts = df["prompt"] + " " + df["response"]
    labels = df["label"].values

    # 🔥 FIT VECTORIZER HERE
    X = vectorizer.fit_transform(texts).toarray()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return X, y


def text_to_vector(text, vectorizer):
    vec = vectorizer.transform([text]).toarray()
    return torch.tensor(vec, dtype=torch.float32)