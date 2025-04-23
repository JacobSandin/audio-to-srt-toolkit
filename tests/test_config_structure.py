import unittest
import os
import sys
import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_toolkit

class TestConfigStructure(unittest.TestCase):
    """Test that the config.yaml file has the correct structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load the actual config file
        self.config = audio_toolkit.load_config()
        
        # Skip tests if no config file exists
        if not self.config:
            self.skipTest("No config.yaml file found, skipping tests")
    
    def test_config_has_audio_processing_section(self):
        """Test that the config file has the audio_processing section."""
        self.assertIn("audio_processing", self.config)
    
    def test_config_has_models_section(self):
        """Test that the config file has the models section."""
        self.assertIn("models", self.config["audio_processing"])
    
    def test_config_has_diarization_section(self):
        """Test that the config file has the diarization section."""
        self.assertIn("diarization", self.config["audio_processing"]["models"])
    
    def test_config_has_vad_section(self):
        """Test that the config file has the vad section."""
        self.assertIn("vad", self.config["audio_processing"]["models"])
    
    def test_config_has_segmentation_section(self):
        """Test that the config file has the segmentation section."""
        self.assertIn("segmentation", self.config["audio_processing"]["models"])
    
    def test_diarization_has_primary_and_fallback(self):
        """Test that the diarization section has primary and fallback."""
        diarization = self.config["audio_processing"]["models"]["diarization"]
        self.assertIn("primary", diarization)
        self.assertIn("fallback", diarization)
    
    def test_vad_has_primary_and_fallback(self):
        """Test that the vad section has primary and fallback."""
        vad = self.config["audio_processing"]["models"]["vad"]
        self.assertIn("primary", vad)
        self.assertIn("fallback", vad)
    
    def test_segmentation_has_primary_and_fallback(self):
        """Test that the segmentation section has primary and fallback."""
        segmentation = self.config["audio_processing"]["models"]["segmentation"]
        self.assertIn("primary", segmentation)
        self.assertIn("fallback", segmentation)
    
    def test_diarization_models_not_empty(self):
        """Test that the diarization models are not empty."""
        diarization = self.config["audio_processing"]["models"]["diarization"]
        self.assertTrue(len(diarization["primary"]) > 0)
        self.assertTrue(len(diarization["fallback"]) > 0)
    
    def test_vad_models_not_empty(self):
        """Test that the vad models are not empty."""
        vad = self.config["audio_processing"]["models"]["vad"]
        self.assertTrue(len(vad["primary"]) > 0)
        self.assertTrue(len(vad["fallback"]) > 0)
    
    def test_segmentation_models_not_empty(self):
        """Test that the segmentation models are not empty."""
        segmentation = self.config["audio_processing"]["models"]["segmentation"]
        self.assertTrue(len(segmentation["primary"]) > 0)
        self.assertTrue(len(segmentation["fallback"]) > 0)
    
    def test_huggingface_token_exists(self):
        """Test that the huggingface token exists."""
        self.assertIn("authentication", self.config)
        self.assertIn("huggingface_token", self.config["authentication"])
        token = self.config["authentication"]["huggingface_token"]
        self.assertTrue(token, "Hugging Face token is empty")

if __name__ == '__main__':
    unittest.main()
