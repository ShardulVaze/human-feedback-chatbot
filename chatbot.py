from groq import Groq
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_responses(user_input):
    responses = []

    # 🔥 Different prompts for diversity (IMPORTANT)
    prompts = [
        f"Explain simply: {user_input}",
        f"Explain in detail: {user_input}",
        f"Give a short and clear answer: {user_input}"
    ]

    for prompt in prompts:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",   
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        response = completion.choices[0].message.content.strip()
        responses.append(response)

    return responses