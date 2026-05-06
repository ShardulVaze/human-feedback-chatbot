import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from reward_model import RewardModel

# Load data
data = pd.read_csv("feedback_data.csv")

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Prepare inputs
X = vectorizer.transform(data["bot_response"]).toarray()
y = data["feedback"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Model
model = RewardModel(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "reward_model.pth")
print("Model trained and saved!")