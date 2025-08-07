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
        self._model_manager = None
        self._initialized = False
        logging.info("InferenceProcessor initialized (lazy loading)")

    @property
    def model_manager(self):
        """Lazy-loaded model manager property"""
        if not self._initialized:
            self._ensure_initialized()
        return self._model_manager

    def _ensure_initialized(self):
        """Ensure the model manager is initialized (lazy loading)"""
        if not self._initialized:
            system_prompt = open("schemas/action_processing/system_prompt.txt", "r").read().strip()
            structured_response = json.load(
                open("schemas/action_processing/structured_output.json", "r")
            )
            self._model_manager = MLModelManager(
                "tiiuae-falcon-40b-instruct", system_prompt, structured_response
            )
            self._initialized = True
            logging.info("InferenceProcessor model manager initialized")

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
        result = Action(**response)
        return result