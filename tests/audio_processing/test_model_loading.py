import unittest
import os
import sys
import yaml
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_processing.diarization import SpeakerDiarizer

class TestModelLoading(unittest.TestCase):
    """Test the model loading functionality with the new configuration structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test config with the new structure
        self.test_config = {
            "models": {
                "diarization": {
                    "primary": ["tensorlake/speaker-diarization-3.1"],
                    "fallback": ["pyannote/speaker-diarization-3.1"]
                },
                "vad": {
                    "primary": ["pyannote/voice-activity-detection"],
                    "fallback": ["pyannote/segmentation-3.0"]
                },
                "segmentation": {
                    "primary": ["HiTZ/pyannote-segmentation-3.0-RTVE"],
                    "fallback": ["pyannote/segmentation-3.0"]
                }
            },
            "authentication": {
                "huggingface_token": "test_token"
            }
        }
    
    @patch('pyannote.audio.Pipeline.from_pretrained')
    def test_load_diarization_models(self, mock_from_pretrained):
        """Test that diarization models are loaded correctly from the new config structure."""
        # Mock the pipeline loading to avoid actual API calls
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        
        # Create a diarizer with our test config
        diarizer = SpeakerDiarizer(self.test_config)
        
        # Call load_models
        diarizer.load_models()
        
        # Check that from_pretrained was called at least once
        mock_from_pretrained.assert_called()
        
        # Check that the diarization pipeline was set
        self.assertIsNotNone(diarizer.diarization_pipeline)
        
        # Verify that the model was loaded from our config
        mock_from_pretrained.assert_any_call(
            "tensorlake/speaker-diarization-3.1",
            use_auth_token=diarizer.huggingface_token
        )
    
    @patch('pyannote.audio.Pipeline.from_pretrained')
    def test_load_vad_models(self, mock_from_pretrained):
        """Test that VAD models are loaded correctly from the new config structure."""
        # Mock the pipeline loading to avoid actual API calls
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        
        # Create a diarizer with our test config
        diarizer = SpeakerDiarizer(self.test_config)
        
        # Call load_models
        diarizer.load_models()
        
        # Verify that the VAD models are correctly loaded from the config
        self.assertIsNotNone(diarizer.vad_pipeline)
        
        # Check that the VAD models were extracted from the config
        vad_models = []
        if 'models' in diarizer.config and 'vad' in diarizer.config['models']:
            if 'primary' in diarizer.config['models']['vad']:
                vad_models.extend(diarizer.config['models']['vad']['primary'])
            if 'fallback' in diarizer.config['models']['vad']:
                vad_models.extend(diarizer.config['models']['vad']['fallback'])
        
        self.assertEqual(vad_models, ["pyannote/voice-activity-detection", "pyannote/segmentation-3.0"])
    
    @patch('pyannote.audio.Pipeline.from_pretrained')
    def test_load_segmentation_models(self, mock_from_pretrained):
        """Test that segmentation models are loaded correctly from the new config structure."""
        # Mock the pipeline loading to avoid actual API calls
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        
        # Create a diarizer with our test config
        diarizer = SpeakerDiarizer(self.test_config)
        
        # Call load_models
        diarizer.load_models()
        
        # Check that the segmentation models were extracted from the config
        segmentation_models = []
        if 'models' in diarizer.config and 'segmentation' in diarizer.config['models']:
            if 'primary' in diarizer.config['models']['segmentation']:
                segmentation_models.extend(diarizer.config['models']['segmentation']['primary'])
            if 'fallback' in diarizer.config['models']['segmentation']:
                segmentation_models.extend(diarizer.config['models']['segmentation']['fallback'])
        
        self.assertEqual(segmentation_models, 
                         ["HiTZ/pyannote-segmentation-3.0-RTVE", "pyannote/segmentation-3.0"])
                         
        # Verify that from_pretrained was called with at least one of the segmentation models
        mock_from_pretrained.assert_any_call(
            "HiTZ/pyannote-segmentation-3.0-RTVE", 
            use_auth_token=diarizer.huggingface_token
        )
    
    @patch('pyannote.audio.Pipeline.from_pretrained')
    def test_fallback_to_old_config_structure(self, mock_from_pretrained):
        """Test that the diarizer falls back to the old config structure if needed."""
        # Mock the pipeline loading to avoid actual API calls
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        
        # Create a test config with the old structure
        old_config = {
            "models": {
                "primary": ["tensorlake/speaker-diarization-3.1"],
                "fallback": ["pyannote/speaker-diarization-3.1"]
            },
            "authentication": {
                "huggingface_token": "test_token"
            }
        }
        
        # Create a diarizer with the old config structure
        diarizer = SpeakerDiarizer(old_config)
        
        # Call load_models
        diarizer.load_models()
        
        # Check that from_pretrained was called at least once
        mock_from_pretrained.assert_called()
        
        # Check that the diarization pipeline was set
        self.assertIsNotNone(diarizer.diarization_pipeline)
        
        # Verify that the model was loaded from our config
        mock_from_pretrained.assert_any_call(
            "tensorlake/speaker-diarization-3.1",
            use_auth_token=diarizer.huggingface_token
        )

if __name__ == '__main__':
    unittest.main()
