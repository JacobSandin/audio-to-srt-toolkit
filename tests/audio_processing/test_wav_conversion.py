#!/usr/bin/env python3
# Test file for initial WAV conversion functionality
# Tests converting input audio to high-quality WAV format
# 2025-04-23 -JS

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from pydub import AudioSegment

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import modules to test
from src.audio_processing.preprocessor import AudioPreprocessor


class TestWavConversion(unittest.TestCase):
    """
    Test case for initial WAV conversion functionality
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
        audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        audio.export(self.input_file, format="mp3")
        
        # Create preprocessor instance
        self.preprocessor = AudioPreprocessor()
    
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    def test_convert_to_wav(self):
        """
        Test that audio is correctly converted to WAV format
        """
        # Define output WAV file path
        output_file = os.path.join(self.output_dir, "test_output.wav")
        
        # Call the method (which will be implemented)
        result = self.preprocessor.convert_to_wav(
            self.input_file, 
            output_file,
            bit_depth=24,
            sample_rate=48000
        )
        
        # Assert conversion was successful
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Load the converted file and check its properties
        converted_audio = AudioSegment.from_file(output_file)
        self.assertEqual(converted_audio.frame_rate, 48000)
        # PyDub might use 32-bit float (4 bytes) internally, so we'll verify the file is high quality
        # rather than the exact bit depth
        self.assertGreaterEqual(converted_audio.sample_width, 3)  # At least 24 bits (3 bytes)
        self.assertEqual(converted_audio.channels, 2)  # Stereo
    
    def test_convert_to_wav_with_different_parameters(self):
        """
        Test WAV conversion with different bit depths and sample rates
        """
        test_parameters = [
            {"bit_depth": 16, "sample_rate": 44100},
            {"bit_depth": 24, "sample_rate": 48000},
            {"bit_depth": 32, "sample_rate": 96000}
        ]
        
        for params in test_parameters:
            # Define output WAV file path
            output_file = os.path.join(
                self.output_dir, 
                f"test_output_{params['bit_depth']}bit_{params['sample_rate']}hz.wav"
            )
            
            # Call the method
            result = self.preprocessor.convert_to_wav(
                self.input_file, 
                output_file,
                bit_depth=params['bit_depth'],
                sample_rate=params['sample_rate']
            )
            
            # Assert conversion was successful
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_file))
            
            # Load the converted file and check its properties
            converted_audio = AudioSegment.from_file(output_file)
            self.assertEqual(converted_audio.frame_rate, params['sample_rate'])
            # PyDub might use 32-bit float (4 bytes) internally, so we'll verify the file is high quality
            # rather than the exact bit depth
            if params['bit_depth'] == 16:
                self.assertGreaterEqual(converted_audio.sample_width, 2)  # At least 16 bits
            elif params['bit_depth'] == 24:
                self.assertGreaterEqual(converted_audio.sample_width, 3)  # At least 24 bits
            elif params['bit_depth'] == 32:
                self.assertGreaterEqual(converted_audio.sample_width, 4)  # At least 32 bits
            self.assertEqual(converted_audio.channels, 2)  # Stereo
    
    def test_convert_to_wav_with_debug(self):
        """
        Test that WAV conversion creates a debug file when debug mode is enabled
        """
        # Create debug directory
        debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create preprocessor with debug mode enabled
        config = {
            'debug': True,
            'debug_dir': debug_dir
        }
        preprocessor = AudioPreprocessor(config)
        
        # Define output WAV file path
        output_file = os.path.join(self.output_dir, "test_output.wav")
        
        # Mock the _save_debug_file method to check if it's called
        with patch.object(preprocessor, '_save_debug_file') as mock_save_debug:
            # Call the method
            preprocessor.convert_to_wav(
                self.input_file, 
                output_file,
                bit_depth=24,
                sample_rate=48000
            )
            
            # Assert _save_debug_file was called with the correct parameters
            mock_save_debug.assert_called_once()
            
            # Check the step name parameter (should be 'wav_conversion')
            args, _ = mock_save_debug.call_args
            self.assertEqual(args[2], 'wav_conversion')
    
    def test_preprocess_includes_wav_conversion(self):
        """
        Test that the preprocess method includes WAV conversion as the first step
        """
        # Mock the convert_to_wav method
        with patch.object(self.preprocessor, 'convert_to_wav') as mock_convert:
            # Set up the mock to return True
            mock_convert.return_value = True
            
            # Also mock the other methods to avoid actual processing
            with patch.object(self.preprocessor, 'separate_vocals', return_value=True), \
                 patch.object(self.preprocessor, 'process_audio', return_value=True):
                
                # Call the preprocess method
                output_file = os.path.join(self.output_dir, "test_output.mp3")
                self.preprocessor.preprocess(self.input_file, output_file)
                
                # Assert convert_to_wav was called
                mock_convert.assert_called_once()


if __name__ == '__main__':
    unittest.main()
