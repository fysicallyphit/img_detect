
from openai import OpenAI
import os
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

response = client.responses.create(
    model = "gpt-4o-mini",
    input = "hello"
)

print(response.output_text)
