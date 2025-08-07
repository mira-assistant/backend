"""
Data Acquisition Script for Mira Assistant Training Datasets

This script identifies and acquires publicly available datasets for:
1. Command-based conversational tasks (for LLaMA-2-7B-Chat)
2. Data extraction tasks (for Falcon-40B-Instruct)

The script downloads, processes, and formats datasets into JSONL format
suitable for fine-tuning the models.
"""

import argparse
import json
import logging
import os
import requests
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetAcquisition:
    """Dataset acquisition and processing for Mira training."""
    
    def __init__(self, output_dir: str):
        """
        Initialize dataset acquisition.
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.command_datasets = {
            "assistant_conversations": {
                "source": "microsoft/DialoGPT-medium",
                "description": "Conversational AI training data",
                "format": "huggingface",
                "task": "command_processing"
            },
            "personal_assistant": {
                "source": "https://raw.githubusercontent.com/microsoft/botframework-solutions/main/samples/csharp/assistants/virtual-assistant/VirtualAssistantSample/VirtualAssistantSample/Responses/MainResponses.json",
                "description": "Virtual assistant response templates",
                "format": "json",
                "task": "command_processing"
            },
            "conversational_qa": {
                "source": "squad",
                "description": "Stanford Question Answering Dataset",
                "format": "huggingface",
                "task": "command_processing"
            }
        }
        
        self.extraction_datasets = {
            "calendar_events": {
                "source": "https://raw.githubusercontent.com/RasaHQ/rasa-nlu-examples/main/data/calendar/train.yml",
                "description": "Calendar event extraction data",
                "format": "yaml",
                "task": "data_extraction"
            },
            "contact_extraction": {
                "source": "conll2003",
                "description": "Named Entity Recognition dataset",
                "format": "huggingface", 
                "task": "data_extraction"
            },
            "action_classification": {
                "source": "mteb/intent-classification",
                "description": "Intent classification dataset",
                "format": "huggingface",
                "task": "data_extraction"
            }
        }
    
    def download_huggingface_dataset(self, dataset_name: str, task_type: str) -> List[Dict[str, Any]]:
        """
        Download dataset from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            task_type: Type of task (command_processing or data_extraction)
            
        Returns:
            List of processed examples
        """
        logger.info(f"Downloading Hugging Face dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name, split='train[:1000]')
            examples = []
            
            if task_type == "command_processing":
                for item in dataset:
                    if dataset_name == "squad":
                        example = {
                            "input": item['question'],
                            "output": item['answers']['text'][0] if item['answers']['text'] else "",
                            "context": item['context'],
                            "task": "question_answering"
                        }
                    else:
                        example = {
                            "input": str(item.get('text', item.get('question', ''))),
                            "output": str(item.get('response', item.get('answer', ''))),
                            "context": "",
                            "task": "conversation"
                        }
                    examples.append(example)
            
            elif task_type == "data_extraction":
                for item in dataset:
                    if dataset_name == "conll2003":
                        tokens = item['tokens']
                        ner_tags = item['ner_tags']
                        text = " ".join(tokens)
                        
                        entities = []
                        current_entity = []
                        current_label = None
                        
                        for token, tag in zip(tokens, ner_tags):
                            if tag != 0:
                                if current_label is None:
                                    current_entity = [token]
                                    current_label = tag
                                else:
                                    current_entity.append(token)
                            else:
                                if current_entity:
                                    entities.append(" ".join(current_entity))
                                    current_entity = []
                                    current_label = None
                        
                        example = {
                            "input": text,
                            "output": json.dumps({"entities": entities}),
                            "context": "",
                            "task": "entity_extraction"
                        }
                    else:
                        example = {
                            "input": str(item.get('text', '')),
                            "output": json.dumps(item.get('label', {})),
                            "context": "",
                            "task": "classification"
                        }
                    examples.append(example)
            
            logger.info(f"Processed {len(examples)} examples from {dataset_name}")
            return examples
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return []
    
    def download_url_dataset(self, url: str, task_type: str) -> List[Dict[str, Any]]:
        """
        Download dataset from URL.
        
        Args:
            url: URL to download from
            task_type: Type of task
            
        Returns:
            List of processed examples
        """
        logger.info(f"Downloading dataset from URL: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            examples = []
            
            if url.endswith('.json'):
                data = response.json()
                if isinstance(data, list):
                    for item in data[:1000]:
                        example = {
                            "input": str(item.get('input', item.get('text', ''))),
                            "output": str(item.get('output', item.get('response', ''))),
                            "context": str(item.get('context', '')),
                            "task": task_type
                        }
                        examples.append(example)
                elif isinstance(data, dict):
                    for key, value in list(data.items())[:1000]:
                        example = {
                            "input": key,
                            "output": str(value),
                            "context": "",
                            "task": task_type
                        }
                        examples.append(example)
            
            logger.info(f"Processed {len(examples)} examples from URL")
            return examples
            
        except Exception as e:
            logger.error(f"Error downloading from URL {url}: {e}")
            return []
    
    def create_synthetic_data(self, task_type: str, count: int = 500) -> List[Dict[str, Any]]:
        """
        Create synthetic training data.
        
        Args:
            task_type: Type of task
            count: Number of synthetic examples to create
            
        Returns:
            List of synthetic examples
        """
        logger.info(f"Creating {count} synthetic examples for {task_type}")
        
        examples = []
        
        if task_type == "command_processing":
            templates = [
                ("What's the weather like?", "Let me check the weather for you."),
                ("What time is it?", "The current time is {time}."),
                ("Set a reminder for tomorrow", "I'll set a reminder for tomorrow."),
                ("Call my mom", "I'll help you call your mom."),
                ("Schedule a meeting", "I'll help you schedule a meeting."),
                ("Turn off the lights", "I'll turn off the lights for you."),
                ("Play some music", "I'll play some music for you."),
                ("Send a text message", "I'll help you send a text message."),
            ]
            
            for i in range(count):
                template = templates[i % len(templates)]
                example = {
                    "input": template[0],
                    "output": template[1],
                    "context": "",
                    "task": "command"
                }
                examples.append(example)
        
        elif task_type == "data_extraction":
            templates = [
                ("Remind me to call John tomorrow at 3pm", 
                 json.dumps({"call_to_action": True, "action_type": "remind", "remind": {"time": "2024-01-01T15:00:00", "description": "call John"}})),
                ("Schedule dinner with Sarah on Friday at 7pm",
                 json.dumps({"call_to_action": True, "action_type": "schedule", "schedule": {"start_time": "2024-01-05T19:00:00", "end_time": "2024-01-05T21:00:00", "name": "dinner with Sarah"}})),
                ("Text my brother that I'm running late",
                 json.dumps({"call_to_action": True, "action_type": "contact", "contact": {"name": "brother", "method": "message", "details": "I'm running late"}})),
                ("I need to pick up groceries after work",
                 json.dumps({"call_to_action": True, "action_type": "remind", "remind": {"time": "2024-01-01T17:00:00", "description": "pick up groceries"}})),
            ]
            
            for i in range(count):
                template = templates[i % len(templates)]
                example = {
                    "input": template[0],
                    "output": template[1],
                    "context": "",
                    "task": "extraction"
                }
                examples.append(example)
        
        logger.info(f"Created {len(examples)} synthetic examples")
        return examples
    
    def save_dataset(self, examples: List[Dict[str, Any]], filename: str):
        """
        Save dataset to JSONL file.
        
        Args:
            examples: List of examples to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def acquire_command_datasets(self):
        """Acquire datasets for command processing task."""
        logger.info("Acquiring command processing datasets")
        
        all_examples = []
        
        for name, info in self.command_datasets.items():
            if info["format"] == "huggingface":
                examples = self.download_huggingface_dataset(info["source"], "command_processing")
            else:
                examples = self.download_url_dataset(info["source"], "command_processing")
            
            all_examples.extend(examples)
        
        synthetic_examples = self.create_synthetic_data("command_processing", 500)
        all_examples.extend(synthetic_examples)
        
        self.save_dataset(all_examples, "command_processing.jsonl")
        
        return len(all_examples)
    
    def acquire_extraction_datasets(self):
        """Acquire datasets for data extraction task.""" 
        logger.info("Acquiring data extraction datasets")
        
        all_examples = []
        
        for name, info in self.extraction_datasets.items():
            if info["format"] == "huggingface":
                examples = self.download_huggingface_dataset(info["source"], "data_extraction")
            else:
                examples = self.download_url_dataset(info["source"], "data_extraction")
            
            all_examples.extend(examples)
        
        synthetic_examples = self.create_synthetic_data("data_extraction", 500)
        all_examples.extend(synthetic_examples)
        
        self.save_dataset(all_examples, "data_extraction.jsonl")
        
        return len(all_examples)


def main():
    """Main function to run data acquisition."""
    parser = argparse.ArgumentParser(description="Acquire training datasets for Mira assistant")
    parser.add_argument("--task", choices=["command", "extraction", "both"], default="both",
                        help="Which datasets to acquire")
    parser.add_argument("--output-dir", default="datasets",
                        help="Output directory for datasets")
    
    args = parser.parse_args()
    
    acquisition = DatasetAcquisition(args.output_dir)
    
    if args.task in ["command", "both"]:
        command_count = acquisition.acquire_command_datasets()
        logger.info(f"Acquired {command_count} command processing examples")
    
    if args.task in ["extraction", "both"]:
        extraction_count = acquisition.acquire_extraction_datasets()
        logger.info(f"Acquired {extraction_count} data extraction examples")
    
    logger.info("Dataset acquisition completed successfully")


if __name__ == "__main__":
    main()