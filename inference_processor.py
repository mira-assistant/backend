import json
import logging
from ml_model_manager import MLModelManager
from models import Action, Interaction

class InferenceProcessor:

    def __init__(self):
        """
        Initialize the inference processor.
        This class is responsible for managing the ML model and processing prompts.
        """

        system_prompt = open("schemas/action_processing/system_prompt.txt", "r").read().strip()
        structured_response = json.load(open("schemas/action_processing/structured_output.json", "r"))
        self.model_manager = MLModelManager("falcon-40b-instruct", system_prompt, structured_response)

        logging.info("InferenceProcessor initialized")


    def extract_action(self, interaction: Interaction, context=None) -> Action:
        """
        Sends a prompt to the LM Studio API for action data extraction.
        This function is maintained for backward compatibility but now uses
        the ML model manager for structured action extraction.

        Args:
            interaction: User interaction object
            context: Optional context information

        Returns:
            Action: Extracted action model
        """

        response = self.model_manager.run_inference(interaction, context)

        result = Action(
            **response
        )

        return result

    @staticmethod
    def send_prompt(prompt: str, context: str = None) -> dict:
        """
        Static method for backward compatibility with existing code.
        
        Args:
            prompt: User prompt text
            context: Optional context information
            
        Returns:
            dict: Response from the model
        """
        # Create a temporary interaction object for the static method
        from models import Interaction
        import uuid
        from datetime import datetime, timezone
        
        temp_interaction = Interaction(
            id=uuid.uuid4(),
            text=prompt,
            timestamp=datetime.now(timezone.utc),
            speaker_id=uuid.uuid4()  # Temporary speaker ID
        )
        
        # Create temporary processor instance
        processor = InferenceProcessor()
        response = processor.model_manager.run_inference(temp_interaction, context)
        
        return response
