#!/usr/bin/env python3
# Test file for diarization integration with audio_toolkit.py
# Tests command-line interface for diarization functionality
# 2025-04-23 - JS

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock
import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import modules to test
from audio_toolkit import process_audio


class TestDiarizationIntegration(unittest.TestCase):
    """
    Test case for diarization integration with audio_toolkit.py
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "test_input.mp3")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a simple test audio file
        with open(self.input_file, 'wb') as f:
            f.write(b'test audio data')
    
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    @patch('src.audio_processing.preprocessor.AudioPreprocessor')
    @patch('src.audio_processing.diarization.SpeakerDiarizer')
    def test_diarization_integration(self, mock_diarizer, mock_preprocessor):
        """
        Test that the CLI properly integrates with the diarizer
        """
        # Setup mocks
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.preprocess.return_value = True
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        mock_diarizer_instance = MagicMock()
        mock_diarizer_instance.diarize.return_value = True
        mock_diarizer.return_value = mock_diarizer_instance
        
        # Create test args
        class Args:
            pass
        
        args = Args()
        args.input_audio = self.input_file
        args.output_dir = self.output_dir
        args.skip_preprocessing = False
        args.diarize = True
        args.min_speakers = 2
        args.max_speakers = 4
        args.clustering_threshold = 0.65
        args.highpass = 150
        args.lowpass = 8000
        args.compression_threshold = -10.0
        args.compression_ratio = 2.0
        args.volume_gain = 3.0
        args.bit_depth = 24
        args.sample_rate = 48000
        args.quiet = False
        args.debug = True
        
        # Call the process_audio function
        result = process_audio(args)
        
        # Assert preprocessing and diarization were called
        self.assertTrue(result)
        mock_preprocessor_instance.preprocess.assert_called_once()
        mock_diarizer_instance.diarize.assert_called_once()
        
        # Check diarization configuration
        diarizer_config = mock_diarizer.call_args[0][0]
        self.assertEqual(diarizer_config['min_speakers'], 2)
        self.assertEqual(diarizer_config['max_speakers'], 4)
        self.assertEqual(diarizer_config['clustering_threshold'], 0.65)
    
    @patch('src.audio_processing.preprocessor.AudioPreprocessor')
    @patch('src.audio_processing.diarization.SpeakerDiarizer')
    def test_diarization_with_custom_parameters(self, mock_diarizer, mock_preprocessor):
        """
        Test diarization with custom parameters
        """
        # Setup mocks
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.preprocess.return_value = True
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        mock_diarizer_instance = MagicMock()
        mock_diarizer_instance.diarize.return_value = True
        mock_diarizer.return_value = mock_diarizer_instance
        
        # Create test args
        class Args:
            pass
        
        args = Args()
        args.input_audio = self.input_file
        args.output_dir = self.output_dir
        args.skip_preprocessing = False
        args.diarize = True
        args.min_speakers = 3
        args.max_speakers = 5
        args.clustering_threshold = 0.7
        args.highpass = 150
        args.lowpass = 8000
        args.compression_threshold = -10.0
        args.compression_ratio = 2.0
        args.volume_gain = 3.0
        args.bit_depth = 24
        args.sample_rate = 48000
        args.quiet = False
        args.debug = False
        
        # Call the process_audio function
        result = process_audio(args)
        
        # Assert preprocessing and diarization were called
        self.assertTrue(result)
        mock_preprocessor_instance.preprocess.assert_called_once()
        mock_diarizer_instance.diarize.assert_called_once()
        
        # Check diarization configuration
        diarizer_config = mock_diarizer.call_args[0][0]
        self.assertEqual(diarizer_config['min_speakers'], 3)
        self.assertEqual(diarizer_config['max_speakers'], 5)
        self.assertEqual(diarizer_config['clustering_threshold'], 0.7)
        self.assertFalse(diarizer_config['debug'])
    
    @patch('src.audio_processing.preprocessor.AudioPreprocessor')
    @patch('src.audio_processing.diarization.SpeakerDiarizer')
    def test_skip_preprocessing_with_diarization(self, mock_diarizer, mock_preprocessor):
        """
        Test skipping preprocessing but still performing diarization
        """
        # Setup mocks
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        mock_diarizer_instance = MagicMock()
        mock_diarizer_instance.diarize.return_value = True
        mock_diarizer.return_value = mock_diarizer_instance
        
        # Create test args
        class Args:
            pass
        
        args = Args()
        args.input_audio = self.input_file
        args.output_dir = self.output_dir
        args.skip_preprocessing = True
        args.diarize = True
        args.min_speakers = 2
        args.max_speakers = 4
        args.clustering_threshold = 0.65
        args.highpass = 150
        args.lowpass = 8000
        args.compression_threshold = -10.0
        args.compression_ratio = 2.0
        args.volume_gain = 3.0
        args.bit_depth = 24
        args.sample_rate = 48000
        args.quiet = False
        args.debug = False
        
        # Call the process_audio function
        result = process_audio(args)
        
        # Assert preprocessing was skipped but diarization was called
        self.assertTrue(result)
        mock_preprocessor_instance.preprocess.assert_not_called()
        mock_diarizer_instance.diarize.assert_called_once()


if __name__ == '__main__':
    unittest.main()
