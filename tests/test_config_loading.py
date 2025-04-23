#!/usr/bin/env python3
# Test file for configuration loading functionality
# Tests that config.yaml is loaded correctly
# 2025-04-23 -JS

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, mock_open, MagicMock
import pytest
import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main module
import audio_toolkit


class TestConfigLoading(unittest.TestCase):
    """
    Test case for the configuration loading functionality
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    def test_load_config_file_exists(self):
        """
        Test that load_config correctly loads configuration from a file
        """
        # Create a mock config file content
        mock_config = {
            'authentication': {
                'huggingface_token': 'test_token'
            },
            'audio_processing': {
                'models': {
                    'diarization': {
                        'primary': ['test-model-1'],
                        'fallback': ['test-model-2']
                    }
                }
            }
        }
        
        # Mock the open function to return our test config
        mock_file = mock_open(read_data=yaml.dump(mock_config))
        
        # Patch os.path.exists to return True and open to return our mock file
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_file):
                # Call the load_config function
                config = audio_toolkit.load_config()
                
                # Verify the config was loaded correctly
                self.assertEqual(config['authentication']['huggingface_token'], 'test_token')
                self.assertEqual(config['audio_processing']['models']['diarization']['primary'][0], 'test-model-1')
                self.assertEqual(config['audio_processing']['models']['diarization']['fallback'][0], 'test-model-2')
    
    def test_load_config_file_not_exists(self):
        """
        Test that load_config returns an empty dict when the file doesn't exist
        """
        # Patch os.path.exists to return False
        with patch('os.path.exists', return_value=False):
            # Call the load_config function
            config = audio_toolkit.load_config()
            
            # Verify an empty dict was returned
            self.assertEqual(config, {})
    
    def test_load_config_invalid_yaml(self):
        """
        Test that load_config handles invalid YAML gracefully
        """
        # Mock the open function to return invalid YAML
        mock_file = mock_open(read_data="invalid: yaml: content: - not valid")
        
        # Patch os.path.exists to return True and open to return our mock file
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_file):
                # Patch yaml.safe_load to raise an exception
                with patch('yaml.safe_load', side_effect=yaml.YAMLError):
                    # Call the load_config function
                    config = audio_toolkit.load_config()
                    
                    # Verify an empty dict was returned
                    self.assertEqual(config, {})


if __name__ == '__main__':
    unittest.main()
