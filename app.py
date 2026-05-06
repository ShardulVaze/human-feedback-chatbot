import streamlit as st
import torch
import pickle
import numpy as np
import pandas as pd

from chatbot import generate_responses
from reward_model import RewardModel

st.title(" RLHF Chatbot")

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model
input_size = len(vectorizer.get_feature_names_out())
model = RewardModel(input_size)
model.load_state_dict(torch.load("reward_model.pth"))
model.eval()

# User input
user_input = st.text_input("Enter your message:")

if user_input:
    responses = generate_responses(f"Explain clearly: {user_input}")

    scores = []
    for response in responses:
        vec = vectorizer.transform([response]).toarray()
        vec = torch.tensor(vec, dtype=torch.float32)

        with torch.no_grad():
            score = model(vec).item()

        scores.append(score)

    best_idx = np.argmax(scores)

    st.subheader("Best Response:")
    st.write(responses[best_idx])

    st.subheader("All Responses:")
    
    for i, (resp, score) in enumerate(zip(responses, scores)):
        st.write(f"Response {i+1}: {resp}")
        st.write(f"Score: {round(score, 3)}")


    st.subheader(" Select the Best Response (Your Feedback)")

    selected = st.radio(
        "Which response is actually the best?",
        options=[f"Response {i+1}" for i in range(len(responses))]
    )

    if st.button("Submit Feedback"):
        selected_idx = int(selected.split()[-1]) - 1

        # Save feedback
        df = pd.DataFrame([{
            "user_input": user_input,
            "bot_response": responses[selected_idx],
            "feedback": 1
        }])

        df.to_csv("feedback_data.csv", mode="a", header=False, index=False)

        st.success("Feedback saved! You are training the model.")