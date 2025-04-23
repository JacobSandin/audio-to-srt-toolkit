#!/usr/bin/env python3
# Test file for model configuration structure
# Tests that diarization.py correctly reads from the new configuration structure
# 2025-04-23 -JS

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import modules to test
from src.audio_processing.diarization import SpeakerDiarizer


class TestModelConfiguration(unittest.TestCase):
    """
    Test case for the new model configuration structure
    """
    
    def test_load_models_new_config_structure(self):
        """
        Test that load_models correctly reads from the new configuration structure
        """
        # Create a config with the new structure
        config = {
            'models': {
                'diarization': {
                    'primary': ['test-diarization-model-1'],
                    'fallback': ['test-diarization-model-2']
                },
                'vad': {
                    'primary': ['test-vad-model-1'],
                    'fallback': ['test-vad-model-2']
                },
                'segmentation': {
                    'primary': ['test-segmentation-model-1'],
                    'fallback': ['test-segmentation-model-2']
                }
            },
            'huggingface_token': 'test_token'
        }
        
        # Create a diarizer with our test config
        with patch('pyannote.audio.Pipeline.from_pretrained') as mock_from_pretrained:
            # Configure the mock to return a MagicMock when called
            mock_pipeline = MagicMock()
            mock_from_pretrained.return_value = mock_pipeline
            
            # Create the diarizer
            diarizer = SpeakerDiarizer(config)
            
            # Call load_models
            result = diarizer.load_models()
            
            # Verify that the correct models were loaded
            self.assertTrue(result)
            
            # Check that from_pretrained was called with the correct models
            # We expect it to be called at least 3 times (once for each model type)
            self.assertGreaterEqual(mock_from_pretrained.call_count, 3)
            
            # Check the first call was with the primary diarization model
            first_call_args = mock_from_pretrained.call_args_list[0][0]
            self.assertEqual(first_call_args[0], 'test-diarization-model-1')
    
    def test_load_models_old_config_structure(self):
        """
        Test that load_models correctly falls back to the old configuration structure
        """
        # Create a config with the old structure
        config = {
            'models': {
                'primary': ['test-model-1'],
                'fallback': ['test-model-2'],
                'additional': [
                    'pyannote/voice-activity-detection',
                    'pyannote/segmentation-3.0'
                ]
            },
            'huggingface_token': 'test_token'
        }
        
        # Create a diarizer with our test config
        with patch('pyannote.audio.Pipeline.from_pretrained') as mock_from_pretrained:
            # Configure the mock to return a MagicMock when called
            mock_pipeline = MagicMock()
            mock_from_pretrained.return_value = mock_pipeline
            
            # Create the diarizer
            diarizer = SpeakerDiarizer(config)
            
            # Call load_models
            result = diarizer.load_models()
            
            # Verify that the correct models were loaded
            self.assertTrue(result)
            
            # Check that from_pretrained was called with the correct models
            # We expect it to be called at least 3 times (once for each model type)
            self.assertGreaterEqual(mock_from_pretrained.call_count, 3)
            
            # Check the first call was with the primary model
            first_call_args = mock_from_pretrained.call_args_list[0][0]
            self.assertEqual(first_call_args[0], 'test-model-1')
    
    def test_load_models_no_config(self):
        """
        Test that load_models uses hardcoded defaults when no config is provided
        """
        # Create a diarizer with an empty config
        with patch('pyannote.audio.Pipeline.from_pretrained') as mock_from_pretrained:
            # Configure the mock to return a MagicMock when called
            mock_pipeline = MagicMock()
            mock_from_pretrained.return_value = mock_pipeline
            
            # Create the diarizer with an empty config
            diarizer = SpeakerDiarizer({})
            
            # Call load_models
            result = diarizer.load_models()
            
            # Verify that the correct models were loaded
            self.assertTrue(result)
            
            # Check that from_pretrained was called with the correct models
            # We expect it to be called at least 3 times (once for each model type)
            self.assertGreaterEqual(mock_from_pretrained.call_count, 3)
            
            # Check the first call was with the default diarization model
            first_call_args = mock_from_pretrained.call_args_list[0][0]
            self.assertEqual(first_call_args[0], 'tensorlake/speaker-diarization-3.1')


if __name__ == '__main__':
    unittest.main()
