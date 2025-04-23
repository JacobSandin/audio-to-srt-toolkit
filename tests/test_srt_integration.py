#!/usr/bin/env python3
# Test file for SRT generation integration with audio_toolkit.py
# Tests command-line interface for SRT generation functionality
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


class TestSRTIntegration(unittest.TestCase):
    """
    Test case for SRT generation integration with audio_toolkit.py
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "test_input.wav")
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
    @patch('src.audio_processing.srt_generator.SRTGenerator')
    def test_srt_generation_integration(self, mock_srt_generator, mock_diarizer, mock_preprocessor):
        """
        Test that the CLI properly integrates with the SRT generator
        """
        # Setup mocks
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.preprocess.return_value = True
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        mock_diarizer_instance = MagicMock()
        mock_diarizer_instance.diarize.return_value = True
        mock_diarizer_instance.get_diarization_result.return_value = [
            {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01", "text": ""},
            {"start": 3.0, "end": 5.5, "speaker": "SPEAKER_02", "text": ""}
        ]
        mock_diarizer.return_value = mock_diarizer_instance
        
        mock_srt_generator_instance = MagicMock()
        mock_srt_generator_instance.generate_srt.return_value = True
        mock_srt_generator_instance.merge_segments.return_value = [
            {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01", "text": ""},
            {"start": 3.0, "end": 5.5, "speaker": "SPEAKER_02", "text": ""}
        ]
        mock_srt_generator.return_value = mock_srt_generator_instance
        
        # Create test args
        class Args:
            pass
        
        args = Args()
        args.input_audio = self.input_file
        args.output_dir = self.output_dir
        args.skip_preprocessing = False
        args.skip_diarization = False
        args.skip_srt = False
        args.include_timestamps = True
        args.speaker_format = "{speaker}:"
        args.max_gap = 1.0
        args.max_duration = 10.0
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
        
        # Assert preprocessing, diarization, and SRT generation were called
        self.assertTrue(result)
        mock_preprocessor_instance.preprocess.assert_called_once()
        mock_diarizer_instance.diarize.assert_called_once()
        mock_diarizer_instance.get_diarization_result.assert_called_once()
        mock_srt_generator_instance.merge_segments.assert_called_once()
        mock_srt_generator_instance.generate_srt.assert_called_once()
        
        # Check SRT generation parameters
        srt_generate_args = mock_srt_generator_instance.generate_srt.call_args[0]
        srt_generate_kwargs = mock_srt_generator_instance.generate_srt.call_args[1]
        
        self.assertEqual(srt_generate_kwargs['speaker_format'], "{speaker}:")
        self.assertEqual(srt_generate_kwargs['include_timestamps'], True)
    
    @patch('src.audio_processing.preprocessor.AudioPreprocessor')
    @patch('src.audio_processing.diarization.SpeakerDiarizer')
    @patch('src.audio_processing.srt_generator.SRTGenerator')
    def test_srt_generation_with_custom_format(self, mock_srt_generator, mock_diarizer, mock_preprocessor):
        """
        Test SRT generation with custom speaker format
        """
        # Setup mocks
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.preprocess.return_value = True
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        mock_diarizer_instance = MagicMock()
        mock_diarizer_instance.diarize.return_value = True
        mock_diarizer_instance.get_diarization_result.return_value = [
            {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01", "text": ""},
            {"start": 3.0, "end": 5.5, "speaker": "SPEAKER_02", "text": ""}
        ]
        mock_diarizer.return_value = mock_diarizer_instance
        
        mock_srt_generator_instance = MagicMock()
        mock_srt_generator_instance.generate_srt.return_value = True
        mock_srt_generator_instance.merge_segments.return_value = [
            {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01", "text": ""},
            {"start": 3.0, "end": 5.5, "speaker": "SPEAKER_02", "text": ""}
        ]
        mock_srt_generator.return_value = mock_srt_generator_instance
        
        # Create test args
        class Args:
            pass
        
        args = Args()
        args.input_audio = self.input_file
        args.output_dir = self.output_dir
        args.skip_preprocessing = False
        args.skip_diarization = False
        args.skip_srt = False
        args.include_timestamps = False
        args.speaker_format = "Person {speaker_id}:"
        args.max_gap = 1.0
        args.max_duration = 10.0
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
        
        # Assert preprocessing, diarization, and SRT generation were called
        self.assertTrue(result)
        mock_preprocessor_instance.preprocess.assert_called_once()
        mock_diarizer_instance.diarize.assert_called_once()
        mock_diarizer_instance.get_diarization_result.assert_called_once()
        mock_srt_generator_instance.merge_segments.assert_called_once()
        mock_srt_generator_instance.generate_srt.assert_called_once()
        
        # Check SRT generation parameters
        srt_generate_kwargs = mock_srt_generator_instance.generate_srt.call_args[1]
        self.assertEqual(srt_generate_kwargs['speaker_format'], "Person {speaker_id}:")
        self.assertEqual(srt_generate_kwargs['include_timestamps'], False)


if __name__ == '__main__':
    unittest.main()
