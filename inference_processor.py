import requests
import json

from ml_model_manager import MLModelManager

# Initialize ML model manager instance for action data extraction
# This will be initialized when first needed
ml_model_manager = None

def get_ml_model_manager():
    """Get or create the ML model manager instance"""
    global ml_model_manager
    if ml_model_manager is None:
        # Note: Model validation will happen when MLModelManager is instantiated
        ml_model_manager = MLModelManager(model_name="microsoft/DialoGPT-small")
    return ml_model_manager

API_URL = "http://localhost:1234/v1/chat/completions"


def send_prompt(prompt: str, context=None) -> dict[str, str]:
    """
    Sends a prompt to the LM Studio API for action data extraction.
    This function is maintained for backward compatibility but now uses
    the ML model manager for structured action extraction.
    
    prompt: str - The input prompt to send to the model.
    context: str - The context to include with the prompt.
    Returns: dict - The generated response from the model.
    """
    # Use ML model manager for action extraction - let exceptions propagate
    manager = get_ml_model_manager()
    response = manager.process_action_extraction(prompt, context)
    
    # Convert to expected format for backward compatibility
    result = {
        "action_type": response.action_type,
        "action_data": response.action_data,
        "user_response": response.user_response
    }
    
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
