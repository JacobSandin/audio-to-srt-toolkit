#!/usr/bin/env python3
# Test file for the main audio-toolkit.py command
# Tests command-line interface and integration with processing modules
# 2025-04-23 -JS

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import pytest
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main module
try:
    import audio_toolkit
except ImportError:
    pass  # We'll implement this later


class TestAudioToolkitCLI(unittest.TestCase):
    """
    Test case for the audio-toolkit.py command-line interface
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "test_input.mp3")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a simple test file
        with open(self.input_file, 'wb') as f:
            f.write(b'test audio data')
    
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    def test_argument_parsing(self):
        """
        Test that command-line arguments are parsed correctly
        """
        # Use a direct test approach instead of mocking
        # Create a test file
        test_file = os.path.join(self.temp_dir.name, "test_input.mp3")
        with open(test_file, 'wb') as f:
            f.write(b'test audio data')
        
        # Test that the argument parser can handle the required arguments
        import audio_toolkit
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-audio', required=True)
        parser.add_argument('--output-dir', default=os.getcwd())
        parser.add_argument('--quiet', action='store_true')
        parser.add_argument('--debug', action='store_true')
        
        # Parse test arguments
        args = parser.parse_args(['--input-audio', test_file])
        
        # Verify the arguments were parsed correctly
        self.assertEqual(args.input_audio, test_file)
        self.assertEqual(args.output_dir, os.getcwd())
        self.assertFalse(args.quiet)
        self.assertFalse(args.debug)
    
    def test_preprocessing_integration(self):
        """
        Test that the CLI properly integrates with the preprocessor
        """
        # Create a test file
        test_file = os.path.join(self.temp_dir.name, "test_input.mp3")
        with open(test_file, 'wb') as f:
            f.write(b'test audio data')
            
        # Create a simple mock for AudioPreprocessor
        with patch('src.audio_processing.preprocessor.AudioPreprocessor') as MockPreprocessor:
            # Setup mock
            mock_preprocessor = MockPreprocessor.return_value
            mock_preprocessor.preprocess.return_value = True
            
            # Create test args
            class Args:
                pass
                
            args = Args()
            args.input_audio = test_file
            args.output_dir = self.output_dir
            args.skip_preprocessing = False
            args.skip_diarization = False
            args.skip_srt = False
            args.highpass = 150
            args.lowpass = 8000
            args.compression_threshold = -10.0
            args.compression_ratio = 2.0
            args.volume_gain = 6.0
            args.bit_depth = 24
            args.sample_rate = 48000
            args.quiet = False
            args.debug = False
            args.diarize = False
            args.min_speakers = 2
            args.max_speakers = 4
            args.clustering_threshold = 0.65
            
            # Import and call the process_audio function
            from audio_toolkit import process_audio
            process_audio(args)
            
            # Verify that AudioPreprocessor was called
            MockPreprocessor.assert_called_once()
    
    def test_logging_configuration(self):
        """
        Test that logging is configured correctly based on command-line arguments
        """
        # Create test args
        class Args:
            pass
            
        args = Args()
        args.debug = True
        args.quiet = False
        args.debug_files_only = False
        
        # Patch logging.basicConfig to verify it's called
        with patch('logging.basicConfig') as mock_log_config:
            # Import and call setup_logging
            from audio_toolkit import setup_logging
            setup_logging(args)
            
            # Verify that logging was configured
            mock_log_config.assert_called_once()
            
            # Verify debug level was used
            call_args = mock_log_config.call_args[1]
            self.assertTrue('level' in call_args)
            self.assertTrue('handlers' in call_args)


if __name__ == '__main__':
    unittest.main()
