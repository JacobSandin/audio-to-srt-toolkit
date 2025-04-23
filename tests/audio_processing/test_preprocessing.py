#!/usr/bin/env python3
# Test file for audio preprocessing functionality
# Tests the preprocessing steps: demucs, normalization, highpass, compression, volume adjustment
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
from audio_processing.preprocessor import AudioPreprocessor


class TestAudioPreprocessing(unittest.TestCase):
    """
    Test case for audio preprocessing functionality
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "test_input.mp3")
        self.output_file = os.path.join(self.temp_dir.name, "test_output.mp3")
        
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
    
    def test_demucs_separation(self):
        """
        Test that demucs vocal separation works correctly
        """
        with patch('audio_processing.preprocessor.AudioPreprocessor._run_demucs') as mock_demucs:
            # Mock the demucs function to return a success status
            mock_demucs.return_value = True
            
            # Call the method
            result = self.preprocessor.separate_vocals(self.input_file, self.output_file)
            
            # Assert demucs was called with correct parameters
            mock_demucs.assert_called_once()
            self.assertTrue(result)
    
    def test_normalization(self):
        """
        Test that audio normalization works correctly
        """
        with patch('audio_processing.preprocessor.AudioPreprocessor._normalize_audio') as mock_normalize:
            # Create a mock audio segment
            mock_audio = MagicMock(spec=AudioSegment)
            mock_normalize.return_value = mock_audio
            
            # Call the method
            result = self.preprocessor.normalize(mock_audio)
            
            # Assert normalization was called
            mock_normalize.assert_called_once_with(mock_audio)
            self.assertEqual(result, mock_audio)
    
    def test_highpass_filter(self):
        """
        Test that highpass filter works correctly
        """
        with patch('audio_processing.preprocessor.AudioPreprocessor._apply_highpass') as mock_highpass:
            # Create a mock audio segment
            mock_audio = MagicMock(spec=AudioSegment)
            mock_highpass.return_value = mock_audio
            
            # Call the method with default cutoff
            result = self.preprocessor.apply_highpass(mock_audio)
            
            # Assert highpass was called with default parameters
            mock_highpass.assert_called_once()
            self.assertEqual(result, mock_audio)
    
    def test_compression(self):
        """
        Test that compression works correctly
        """
        with patch('audio_processing.preprocessor.AudioPreprocessor._apply_compression') as mock_compress:
            # Create a mock audio segment
            mock_audio = MagicMock(spec=AudioSegment)
            mock_compress.return_value = mock_audio
            
            # Call the method
            result = self.preprocessor.apply_compression(mock_audio)
            
            # Assert compression was called
            mock_compress.assert_called_once_with(mock_audio)
            self.assertEqual(result, mock_audio)
    
    def test_volume_adjustment(self):
        """
        Test that volume adjustment works correctly
        """
        with patch('audio_processing.preprocessor.AudioPreprocessor._adjust_volume') as mock_adjust:
            # Create a mock audio segment
            mock_audio = MagicMock(spec=AudioSegment)
            mock_adjust.return_value = mock_audio
            
            # Call the method with +6dB gain
            result = self.preprocessor.adjust_volume(mock_audio, gain_db=6)
            
            # Assert volume adjustment was called with correct gain
            mock_adjust.assert_called_once_with(mock_audio, 6)
            self.assertEqual(result, mock_audio)
    
    def test_full_preprocessing_pipeline(self):
        """
        Test the complete preprocessing pipeline
        """
        with patch('audio_processing.preprocessor.AudioPreprocessor.separate_vocals') as mock_separate, \
             patch('audio_processing.preprocessor.AudioPreprocessor.process_audio') as mock_process:
            
            # Mock the return values
            mock_separate.return_value = True
            mock_process.return_value = True
            
            # Call the full preprocessing pipeline
            result = self.preprocessor.preprocess(self.input_file, self.output_file)
            
            # Assert both methods were called
            mock_separate.assert_called_once()
            mock_process.assert_called_once()
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
