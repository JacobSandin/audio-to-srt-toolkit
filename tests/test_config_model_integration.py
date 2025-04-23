import unittest
import os
import sys
import yaml
import logging
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_toolkit
from src.audio_processing.diarization import SpeakerDiarizer

class TestConfigModelIntegration(unittest.TestCase):
    """
    Integration test for loading models from the actual config.yaml file.
    
    This test verifies that the real config file can be loaded and used
    to initialize the diarization models correctly.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Configure logging to avoid cluttering test output
        logging.basicConfig(level=logging.ERROR)
        
        # Load the actual config file
        cls.config = audio_toolkit.load_config()
        
        # Skip tests if no config file exists
        if not cls.config:
            raise unittest.SkipTest("No config.yaml file found, skipping integration tests")
    
    def test_config_has_new_structure(self):
        """Test that the config file has the new structure."""
        # Check that the audio_processing section exists
        self.assertIn("audio_processing", self.config)
        
        # Check that the models section exists
        self.assertIn("models", self.config["audio_processing"])
        
        # Check that the diarization section exists
        self.assertIn("diarization", self.config["audio_processing"]["models"])
        
        # Check that the vad section exists
        self.assertIn("vad", self.config["audio_processing"]["models"])
        
        # Check that the segmentation section exists
        self.assertIn("segmentation", self.config["audio_processing"]["models"])
    
    def test_diarization_models_structure(self):
        """Test that the diarization models section has the correct structure."""
        diarization = self.config["audio_processing"]["models"]["diarization"]
        
        # Check that primary and fallback sections exist
        self.assertIn("primary", diarization)
        self.assertIn("fallback", diarization)
        
        # Check that primary contains at least one model
        self.assertTrue(len(diarization["primary"]) > 0)
        
        # Check that fallback contains at least one model
        self.assertTrue(len(diarization["fallback"]) > 0)
    
    def test_vad_models_structure(self):
        """Test that the VAD models section has the correct structure."""
        vad = self.config["audio_processing"]["models"]["vad"]
        
        # Check that primary and fallback sections exist
        self.assertIn("primary", vad)
        self.assertIn("fallback", vad)
        
        # Check that primary contains at least one model
        self.assertTrue(len(vad["primary"]) > 0)
        
        # Check that fallback contains at least one model
        self.assertTrue(len(vad["fallback"]) > 0)
    
    def test_segmentation_models_structure(self):
        """Test that the segmentation models section has the correct structure."""
        segmentation = self.config["audio_processing"]["models"]["segmentation"]
        
        # Check that primary and fallback sections exist
        self.assertIn("primary", segmentation)
        self.assertIn("fallback", segmentation)
        
        # Check that primary contains at least one model
        self.assertTrue(len(segmentation["primary"]) > 0)
        
        # Check that fallback contains at least one model
        self.assertTrue(len(segmentation["fallback"]) > 0)
    
    @patch('pyannote.audio.Pipeline.from_pretrained')
    def test_initialize_diarizer_with_config(self, mock_from_pretrained):
        """Test that the diarizer can be initialized with the config."""
        # Mock the pipeline loading to avoid actual API calls
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        
        # Create a diarizer with the actual config
        diarizer = SpeakerDiarizer(self.config)
        
        # Call load_models
        diarizer.load_models()
        
        # Check that from_pretrained was called at least once
        mock_from_pretrained.assert_called()
        
        # Extract models from config to verify they match the expected structure
        vad_models = []
        if 'audio_processing' in self.config and 'models' in self.config['audio_processing']:
            config_models = self.config['audio_processing']['models']
            if 'vad' in config_models:
                if 'primary' in config_models['vad']:
                    vad_models.extend(config_models['vad']['primary'])
                if 'fallback' in config_models['vad']:
                    vad_models.extend(config_models['vad']['fallback'])
        
        # Check that the config has VAD models
        self.assertTrue(len(vad_models) > 0, "No VAD models found in config")

if __name__ == '__main__':
    unittest.main()
