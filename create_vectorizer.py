import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
data = pd.read_csv("feedback_data.csv")

# Create vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(data["bot_response"])

# Save it
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("vectorizer.pkl created!")