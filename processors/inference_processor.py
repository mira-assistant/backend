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
        self.model_manager = None
        self._initialized = False
        logging.info("InferenceProcessor initialized (lazy loading)")

    def _ensure_initialized(self):
        """Ensure the model manager is initialized (lazy loading)"""
        if not self._initialized:
            try:
                system_prompt = open("schemas/action_processing/system_prompt.txt", "r").read().strip()
                structured_response = json.load(
                    open("schemas/action_processing/structured_output.json", "r")
                )
                self.model_manager = MLModelManager(
                    "tiiuae-falcon-40b-instruct", system_prompt, structured_response
                )
                self._initialized = True
                logging.info("InferenceProcessor model manager initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize InferenceProcessor model manager: {e}")
                self.model_manager = None

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
        
        self._ensure_initialized()
        
        if self.model_manager is None:
            # Return fallback action when model manager is not available
            return Action(
                type="error",
                description="ML inference not available",
                parameters={}
            )

        try:
            response = self.model_manager.run_inference(interaction, context)
            result = Action(**response)
            return result
        except Exception as e:
            logging.warning(f"Action extraction failed: {e}")
            return Action(
                type="error", 
                description=f"Action extraction failed: {e}",
                parameters={}
            )