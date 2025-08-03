import requests
import json

API_URL = "http://localhost:1234/v1/chat/completions"


def send_prompt(prompt: str, context=None) -> dict[str, str]:
    """
    Sends a prompt to the LM Studio API and returns the generated response.
    prompt: str - The input prompt to send to the model.
    context: str - The context to include with the prompt.
    Returns: dict - The generated response from the model.
    """

    if context:
        prompt = f"Current Prompt:\n{prompt}\n\nContext:\n{context}"

    message = {"role": "user", "content": prompt}

    payload = {
        "model": "microsoft/DialoGPT-small",  # Lightweight model good for NLP
        "messages": [message],
        "max_tokens": -1,
        "stream": False,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40,
        "repeat_penalty": 1.2,
        "min_p": 0.2,
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    result = json.loads(generated_text)
    return result


def main():
    print("LM Studio Interactive Client. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        response = send_prompt(user_input)
        print(f"LM Studio: {response}\n")
        print(response.get("call_to_action", "No call to action provided."))


if __name__ == "__main__":
    main()
