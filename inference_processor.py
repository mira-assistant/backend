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
        self.model_manager = MLModelManager("nous-hermes-2-mistral-7b-dpo", system_prompt, structured_response)

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
