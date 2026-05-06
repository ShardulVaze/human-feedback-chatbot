# RLHF-Based AI Chatbot

An AI chatbot that demonstrates Reinforcement Learning from Human Feedback (RLHF) concepts using PyTorch and Groq API.

## Features

- Generates multiple AI responses
- Scores responses using a reward model
- Allows human feedback selection
- Trains reward model from feedback data
- Interactive UI using Streamlit

## Tech Stack

- Python
- Streamlit
- PyTorch
- Scikit-learn
- Pandas
- Groq API

## Project Structure

```bash
app.py                 # Streamlit frontend
chatbot.py             # Response generation
reward_model.py        # Reward model architecture
train_reward.py        # Reward model training
create_vectorizer.py   # TF-IDF vectorizer creation