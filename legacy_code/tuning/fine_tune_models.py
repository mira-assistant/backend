#!/usr/bin/env python3
"""
Fine-tuning Framework Setup for Mira Assistant Models

This script provides a framework for fine-tuning LLaMA-2-7B-Chat and Falcon-40B-Instruct
models using Hugging Face transformers and LoRA (Low-Rank Adaptation) techniques.

Usage:
    python fine_tune_models.py --model llama-2-7b-chat-hf-function-calling-v3 --dataset datasets/command_processing.jsonl
    python fine_tune_models.py --model tiiuae-falcon-40b-instruct --dataset datasets/data_extraction.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFineTuner:
    """Fine-tuning framework for Mira assistant models."""

    def __init__(self, model_name: str, config_path: str):
        """
        Initialize the fine-tuner.

        Args:
            model_name: Name of the model to fine-tune
            config_path: Path to model configuration file
        """
        self.model_name = model_name
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def setup_model_and_tokenizer(self, base_model_path: str):
        """
        Setup the base model and tokenizer.

        Args:
            base_model_path: Path or name of the base model
        """
        logger.info(f"Loading tokenizer and model for {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **self.config["fine_tuning"]["lora_config"]
        )

        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)

        logger.info("Model and tokenizer setup complete")

    def prepare_dataset(self, dataset_path: str) -> Dataset:
        """
        Prepare training dataset from JSONL file.

        Args:
            dataset_path: Path to JSONL dataset file

        Returns:
            Prepared dataset for training
        """
        logger.info(f"Loading dataset from {dataset_path}")

        # Load dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')

        # Tokenize dataset
        def tokenize_function(examples):
            # Format input based on model type
            if self.model_name == "llama-2-7b-chat-hf-function-calling-v3":
                formatted_texts = []
                for i in range(len(examples['input'])):
                    if 'context' in examples and examples['context'][i]:
                        text = self.config["prompt_templates"]["system_with_context"].format(
                            context=examples['context'][i],
                            input=examples['input'][i]
                        )
                    else:
                        text = self.config["prompt_templates"]["command_processing"].format(
                            input=examples['input'][i]
                        )
                    if 'output' in examples:
                        text += examples['output'][i]
                    formatted_texts.append(text)
            else:  # tiiuae-falcon-40b-instruct
                formatted_texts = []
                for i in range(len(examples['input'])):
                    text = self.config["prompt_templates"]["data_extraction"].format(
                        context=examples.get('context', [''])[i] or '',
                        input=examples['input'][i]
                    )
                    if 'output' in examples:
                        text += examples['output'][i]
                    formatted_texts.append(text)

            if not formatted_texts:
                return None

            if self.tokenizer is None:
                raise ValueError("Tokenizer is not initialized")

            tokenized = self.tokenizer(
                formatted_texts,
                truncation=True,
                max_length=self.config["dataset_requirements"]["max_length"],
                padding=False,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        logger.info(f"Dataset prepared with {len(tokenized_dataset)} samples")  # pyright: ignore[reportArgumentType]
        return tokenized_dataset  # pyright: ignore[reportReturnType]

    def train(self, dataset: Dataset, output_dir: str):
        """
        Fine-tune the model.

        Args:
            dataset: Prepared training dataset
            output_dir: Directory to save the fine-tuned model
        """
        logger.info("Starting fine-tuning process")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            **self.config["fine_tuning"]["training_config"],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized")

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Start training
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Fine-tuning complete. Model saved to {output_dir}")


def main():
    """Main function to run fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Mira assistant models")
    parser.add_argument("--model", required=True, choices=["llama-2-7b-chat-hf-function-calling-v3", "tiiuae-falcon-40b-instruct"],
                        help="Model to fine-tune")
    parser.add_argument("--dataset", required=True, help="Path to training dataset (JSONL format)")
    parser.add_argument("--base-model", help="Path or name of base model (if different from model name)")
    parser.add_argument("--output-dir", help="Output directory for fine-tuned model")

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    config_path = base_dir / args.model / "model_config.json"
    output_dir = args.output_dir or str(base_dir / args.model / "fine_tuned")
    base_model = args.base_model or args.model

    # Create fine-tuner and run training
    fine_tuner = ModelFineTuner(args.model, str(config_path))
    fine_tuner.setup_model_and_tokenizer(base_model)

    dataset = fine_tuner.prepare_dataset(args.dataset)
    fine_tuner.train(dataset, output_dir)

    logger.info("Fine-tuning process completed successfully")


if __name__ == "__main__":
    main()