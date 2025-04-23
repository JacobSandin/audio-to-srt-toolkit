#!/usr/bin/env python3
# Test script to verify model loading with the new config structure
# 2025-04-23 - JS

import os
import sys
import logging
import yaml
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.audio_processing.diarization import SpeakerDiarizer

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_config():
    """Load configuration from config.yaml file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return {}

def test_model_loading():
    """Test loading models with the new config structure."""
    logging.info("Testing model loading with the new config structure...")
    
    # Load config
    config = load_config()
    if not config:
        logging.error("No config loaded, aborting test")
        return False
    
    # Create diarizer
    diarizer = SpeakerDiarizer(config)
    
    # Try to load models
    logging.info("Attempting to load models...")
    success = diarizer.load_models()
    
    if success:
        logging.info("✅ Models loaded successfully!")
        return True
    else:
        logging.error("❌ Failed to load models")
        return False

if __name__ == "__main__":
    result = test_model_loading()
    sys.exit(0 if result else 1)
