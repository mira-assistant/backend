#!/usr/bin/env python3
"""
Validation script for Mira Assistant tuning system

This script validates that:
1. Role-based system is working with structured context separation
2. Model configurations are properly loaded
3. LM Studio server configuration is accessible
4. Dataset acquisition system is functional
"""

import json
import sys
from pathlib import Path
import tempfile
import os

def test_role_based_system():
    """Test the role-based system implementation."""
    print("Testing role-based system...")
    
    try:
        from ml_model_manager import MLModelManager
        from models import Interaction
        import uuid
        from datetime import datetime, timezone
        
        # Create a test interaction
        interaction = Interaction(
            id=uuid.uuid4(),
            text="What time is it?",
            timestamp=datetime.now(timezone.utc),
            speaker_id=uuid.uuid4()
        )
        
        # Test that ML Model Manager can handle context with role separation
        # Note: This will fail with actual LM Studio connection, but tests the interface
        try:
            model_manager = MLModelManager(
                model_name="test-model",
                system_prompt="Test prompt",
                temperature=0.7,
                max_tokens=512,
                top_k=40
            )
            print("✗ Model manager creation should fail without proper model setup")
            return False
        except ValueError as e:
            if "not available or loaded" in str(e):
                print("✓ Model validation working correctly")
            else:
                print(f"✗ Unexpected error: {e}")
                return False
        
        print("✓ Role-based system interface validated")
        return True
        
    except Exception as e:
        print(f"✗ Role-based system test failed: {e}")
        return False

def test_model_configurations():
    """Test that model configuration files are valid."""
    print("\nTesting model configurations...")
    
    try:
        # Test LM Studio server config
        server_config_path = Path("tuning/configs/lm_studio_server_config.json")
        if not server_config_path.exists():
            print("✗ LM Studio server config not found")
            return False
        
        with open(server_config_path) as f:
            server_config = json.load(f)
        
        required_keys = ["server_config", "model_configs", "task_specific_configs"]
        for key in required_keys:
            if key not in server_config:
                print(f"✗ Missing key in server config: {key}")
                return False
        
        print("✓ LM Studio server config is valid")
        
        # Test model-specific configs
        models = ["llama-2-7b-chat", "falcon-40b-instruct"]
        for model in models:
            config_path = Path(f"tuning/{model}/model_config.json")
            if not config_path.exists():
                print(f"✗ Model config not found: {model}")
                return False
            
            with open(config_path) as f:
                model_config = json.load(f)
            
            required_sections = ["model_info", "fine_tuning", "inference_config", "prompt_templates"]
            for section in required_sections:
                if section not in model_config:
                    print(f"✗ Missing section in {model} config: {section}")
                    return False
            
            print(f"✓ {model} config is valid")
        
        return True
        
    except Exception as e:
        print(f"✗ Model configuration test failed: {e}")
        return False

def test_tuning_scripts():
    """Test that tuning scripts are executable and have correct structure."""
    print("\nTesting tuning scripts...")
    
    try:
        scripts = [
            "tuning/fine_tune_models.py",
            "tuning/acquire_datasets.py"
        ]
        
        for script_path in scripts:
            if not Path(script_path).exists():
                print(f"✗ Script not found: {script_path}")
                return False
            
            # Check if script is executable
            if not os.access(script_path, os.X_OK):
                print(f"✗ Script not executable: {script_path}")
                return False
            
            print(f"✓ {script_path} exists and is executable")
        
        return True
        
    except Exception as e:
        print(f"✗ Tuning scripts test failed: {e}")
        return False

def test_dataset_acquisition():
    """Test dataset acquisition functionality."""
    print("\nTesting dataset acquisition...")
    
    try:
        # Import the acquisition module to test imports
        sys.path.append('tuning')
        from acquire_datasets import DatasetAcquisition
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            acquisition = DatasetAcquisition(temp_dir)
            
            # Test synthetic data generation
            command_examples = acquisition.create_synthetic_data("command_processing", 10)
            extraction_examples = acquisition.create_synthetic_data("data_extraction", 10)
            
            if len(command_examples) != 10:
                print(f"✗ Expected 10 command examples, got {len(command_examples)}")
                return False
            
            if len(extraction_examples) != 10:
                print(f"✗ Expected 10 extraction examples, got {len(extraction_examples)}")
                return False
            
            # Validate example structure
            required_keys = ["input", "output", "context", "task"]
            for example in command_examples[:1]:  # Check first example
                for key in required_keys:
                    if key not in example:
                        print(f"✗ Missing key in command example: {key}")
                        return False
            
            for example in extraction_examples[:1]:  # Check first example
                for key in required_keys:
                    if key not in example:
                        print(f"✗ Missing key in extraction example: {key}")
                        return False
            
            print("✓ Synthetic data generation working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset acquisition test failed: {e}")
        return False

def test_inference_processor_integration():
    """Test inference processor integration with role-based system."""
    print("\nTesting inference processor integration...")
    
    try:
        from inference_processor import InferenceProcessor
        
        # Test that static method exists
        if not hasattr(InferenceProcessor, 'send_prompt'):
            print("✗ InferenceProcessor.send_prompt method not found")
            return False
        
        print("✓ InferenceProcessor backward compatibility maintained")
        
        # Test that the class can be instantiated (will fail on model loading, but interface should work)
        try:
            processor = InferenceProcessor()
            print("✗ InferenceProcessor should fail without proper model setup")
            return False
        except ValueError as e:
            if "not available or loaded" in str(e):
                print("✓ InferenceProcessor model validation working")
            else:
                print(f"✗ Unexpected error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Inference processor integration test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Mira Assistant Tuning System Validation")
    print("=" * 50)
    
    tests = [
        test_role_based_system,
        test_model_configurations,
        test_tuning_scripts,
        test_dataset_acquisition,
        test_inference_processor_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Tuning system is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())