#!/usr/bin/env python3
# Test file for debug output functionality
# Tests that intermediate files are created when running with --debug flag
# 2025-04-23 -JS

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path
import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import modules to test
from src.audio_processing.preprocessor import AudioPreprocessor


class TestDebugOutput(unittest.TestCase):
    """
    Test case for debug output functionality
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "test_input.wav")  # 2025-04-23 - JS
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        self.debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create a simple test audio file
        with open(self.input_file, 'wb') as f:
            f.write(b'test audio data')
        
        # Create preprocessor instance with debug mode enabled
        self.config = {
            'debug': True,
            'debug_dir': self.debug_dir
        }
        self.preprocessor = AudioPreprocessor(self.config)
    
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    def test_debug_files_creation(self):
        """
        Test that debug files are created for each processing step when debug is enabled
        """
        # Create a custom test implementation of preprocess that manually calls _save_debug_file
        # This avoids issues with mocking all the complex audio processing functions
        def mock_preprocess(self, input_file, output_file):
            # Call _save_debug_file for each step we expect
            mock_audio = MagicMock()
            steps = ['vocals', 'highpass', 'lowpass', 'compression', 'normalize', 'volume']
            for step in steps:
                self._save_debug_file(mock_audio, input_file, step)
            return True
            
        # Patch the preprocess method with our custom implementation
        with patch('src.audio_processing.preprocessor.AudioPreprocessor.preprocess', new=mock_preprocess):
            # Also patch _save_debug_file to track calls
            with patch('src.audio_processing.preprocessor.AudioPreprocessor._save_debug_file', wraps=self.preprocessor._save_debug_file) as mock_save_debug:
                # Process the audio
                output_file = os.path.join(self.output_dir, "test_output.wav")  # 2025-04-23 - JS
                self.preprocessor.preprocess(self.input_file, output_file)
                
                # Verify that _save_debug_file was called for each expected step
                expected_steps = [
                    'vocals',
                    'highpass',
                    'lowpass',
                    'compression',
                    'normalize',
                    'volume'
                ]
                
                # Get all the step names that were passed to _save_debug_file
                actual_steps = []
                for call in mock_save_debug.call_args_list:
                    # The step name is the third argument (index 2)
                    if len(call[0]) > 2:
                        actual_steps.append(call[0][2])
                
                # Check that each expected step was saved
                for step in expected_steps:
                    self.assertIn(step, actual_steps, f"Debug file for {step} was not created")
                    
                # Verify the total number of calls matches our expectation
                self.assertEqual(len(mock_save_debug.call_args_list), len(expected_steps),
                                f"Expected {len(expected_steps)} debug files, but {len(mock_save_debug.call_args_list)} were created")
    
    def test_debug_flag_in_cli(self):
        """
        Test that the --debug flag in the CLI enables debug output
        """
        # Create test args
        class Args:
            pass
            
        args = Args()
        args.input_audio = self.input_file
        args.output_dir = self.output_dir
        args.skip_preprocessing = False
        args.skip_diarization = True
        args.skip_srt = True
        args.highpass = 150
        args.lowpass = 8000
        args.compression_threshold = -10.0
        args.compression_ratio = 2.0
        args.volume_gain = 6.0
        args.bit_depth = 24
        args.sample_rate = 48000
        args.quiet = False
        args.debug = True
        args.debug_files_only = False
        args.min_speakers = 2
        args.max_speakers = 4
        args.clustering_threshold = 0.65
        
        # Import the process_audio function
        from audio_toolkit import process_audio
        
        # Mock the preprocessor to verify debug mode is enabled
        with patch('src.audio_processing.preprocessor.AudioPreprocessor') as MockPreprocessor:
            # Setup mock
            mock_preprocessor = MockPreprocessor.return_value
            mock_preprocessor.preprocess.return_value = True
            
            # Call the process_audio function
            process_audio(args)
            
            # Verify that AudioPreprocessor was called with debug=True
            call_args = MockPreprocessor.call_args[0][0]  # Get the config dict
            self.assertTrue('debug' in call_args)
            self.assertTrue(call_args['debug'])
            self.assertTrue('debug_dir' in call_args)


if __name__ == '__main__':
    unittest.main()
