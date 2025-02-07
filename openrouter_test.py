import os
from openai import OpenAI

# Initialize client with your OpenRouter credentials
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def inspect_response_structure():
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:nitro",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 17*23? Show your reasoning first."}
            ],
            temperature=0.7,
            max_tokens=200
        )

        # Get the message object from the first choice
        message = response.choices[0].message
        
        print("=== Response Structure ===")
        print(f"Type of message object: {type(message)}")
        print(f"All attributes available: {dir(message)}")
        
        print("\n=== Content Details ===")
        print(f"Content type: {type(message.content)}")
        print(f"Content value: {message.content}")
        
        print("\n=== Full Message Object ===")
        print(message)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    inspect_response_structure()