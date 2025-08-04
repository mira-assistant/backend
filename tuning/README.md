# Mira Assistant Model Tuning System

This directory contains the fine-tuning framework and configuration files for the Mira Assistant's AI models.

## Directory Structure

```
tuning/
├── configs/
│   └── lm_studio_server_config.json     # LM Studio server configuration
├── llama-2-7b-chat/
│   └── model_config.json                # Configuration for LLaMA-2-7B-Chat model
├── falcon-40b-instruct/
│   └── model_config.json                # Configuration for Falcon-40B-Instruct model
├── datasets/                            # Generated training datasets
├── fine_tune_models.py                  # Fine-tuning framework script
├── acquire_datasets.py                  # Dataset acquisition script
└── README.md                           # This file
```

## Models and Tasks

### LLaMA-2-7B-Chat
- **Task**: Command Processing
- **Use Case**: Processing user commands, wake word responses, tool function calls
- **Configuration**: `llama-2-7b-chat/model_config.json`
- **Optimized for**: Conversational interactions, function calling, real-time responses

### Falcon-40B-Instruct  
- **Task**: Data Extraction
- **Use Case**: Extracting structured data from user speech (contacts, reminders, calendar events)
- **Configuration**: `falcon-40b-instruct/model_config.json`
- **Optimized for**: Structured output generation, entity extraction, JSON formatting

## LM Studio Server Configuration

The `configs/lm_studio_server_config.json` file contains important server payload configuration entries:

- **Server Settings**: Base URL, API key, timeout, retry logic
- **Model-Specific Parameters**: Temperature, max tokens, top_k, top_p settings
- **Task-Specific Configs**: Optimized settings for command processing vs data extraction

### Key Configuration Options

```json
{
  "temperature": 0.7,           // Controls randomness (0.0-2.0)
  "max_tokens": 2048,           // Maximum response length
  "top_k": 40,                  // Top-k sampling parameter
  "top_p": 0.9,                 // Nucleus sampling threshold
  "repetition_penalty": 1.1,    // Penalty for repetitive text
  "frequency_penalty": 0.0,     // Frequency-based penalty
  "presence_penalty": 0.0,      // Presence-based penalty
  "stop_sequences": ["</s>"]    // Sequences that stop generation
}
```

## Fine-Tuning Framework

### Setup Requirements

```bash
pip install torch transformers datasets peft accelerate
```

### Dataset Acquisition

Generate training datasets for both models:

```bash
# Acquire all datasets
python acquire_datasets.py --task both --output-dir datasets/

# Acquire specific task datasets
python acquire_datasets.py --task command --output-dir datasets/
python acquire_datasets.py --task extraction --output-dir datasets/
```

This script will:
- Download publicly available conversational datasets
- Create synthetic command and extraction examples
- Format data into JSONL files for training

### Fine-Tuning Models

Fine-tune models using LoRA (Low-Rank Adaptation):

```bash
# Fine-tune LLaMA-2-7B-Chat for command processing
python fine_tune_models.py \
  --model llama-2-7b-chat \
  --dataset datasets/command_processing.jsonl \
  --output-dir llama-2-7b-chat/fine_tuned

# Fine-tune Falcon-40B-Instruct for data extraction  
python fine_tune_models.py \
  --model falcon-40b-instruct \
  --dataset datasets/data_extraction.jsonl \
  --output-dir falcon-40b-instruct/fine_tuned
```

### LoRA Configuration

Both models use LoRA for efficient fine-tuning:

**LLaMA-2-7B-Chat LoRA Settings**:
- Rank (r): 16
- Alpha: 32
- Target modules: q_proj, v_proj, k_proj, o_proj
- Dropout: 0.1

**Falcon-40B-Instruct LoRA Settings**:
- Rank (r): 8
- Alpha: 16  
- Target modules: query_key_value, dense
- Dropout: 0.05

## Dataset Requirements

### Command Processing Dataset
- **Tasks**: Weather queries, time requests, system control, calendar management
- **Format**: Conversational JSONL with input/output pairs
- **Minimum Samples**: 1000
- **Max Length**: 512 tokens

### Data Extraction Dataset  
- **Tasks**: Contact requests, reminder extraction, calendar events, action classification
- **Format**: Instruction JSONL with structured JSON outputs
- **Minimum Samples**: 2000
- **Max Length**: 1024 tokens

## Integration with Mira Backend

The tuning system integrates with the main Mira backend through:

1. **ML Model Manager**: Loads configuration from `configs/lm_studio_server_config.json`
2. **Command Processor**: Uses fine-tuned LLaMA-2-7B-Chat model
3. **Inference Processor**: Uses fine-tuned Falcon-40B-Instruct model
4. **Role-Based System**: Supports structured context separation with role attributes

## Monitoring and Evaluation

Training progress and model performance can be monitored through:
- Hugging Face Transformers logging
- TensorBoard integration
- Custom evaluation metrics for task-specific performance
- Integration testing with the main Mira backend

## Best Practices

1. **Start with small datasets** for initial experimentation
2. **Monitor GPU memory usage** during training
3. **Use gradient accumulation** for larger effective batch sizes
4. **Save checkpoints regularly** during long training runs
5. **Validate models** with the main Mira system before deployment