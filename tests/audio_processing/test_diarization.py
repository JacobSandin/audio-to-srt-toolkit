#!/usr/bin/env python3
# Test file for speaker diarization functionality
# Tests multi-stage diarization with Swedish dialect optimization
# 2025-04-23 - JS

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import torch
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import modules to test
from src.audio_processing.diarization import SpeakerDiarizer


class TestSpeakerDiarization(unittest.TestCase):
    """
    Test case for speaker diarization functionality
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "test_input.wav")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a simple test audio file (1 second of silence)
        with open(self.input_file, 'wb') as f:
            # Write a minimal WAV header
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        
        # Create configuration with test values
        self.config = {
            'debug': True,
            'debug_dir': os.path.join(self.output_dir, "debug"),
            'min_speakers': 2,
            'max_speakers': 4,
            'clustering_threshold': 0.65,
            'use_gpu': False,  # Don't use GPU for tests
            'huggingface_token': 'test_token',
            'batch_size': 16
        }
        
        # Create debug directory
        os.makedirs(self.config['debug_dir'], exist_ok=True)
        
        # Mock the PyAnnote pipeline
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value = MagicMock()
        
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    @patch('src.audio_processing.diarization.Pipeline')
    def test_init_diarizer(self, mock_pipeline):
        """
        Test initializing the diarizer with proper configuration
        """
        # Create diarizer
        diarizer = SpeakerDiarizer(self.config)
        
        # Assert configuration was properly set
        self.assertEqual(diarizer.min_speakers, 2)
        self.assertEqual(diarizer.max_speakers, 4)
        self.assertEqual(diarizer.clustering_threshold, 0.65)
        self.assertEqual(diarizer.huggingface_token, 'test_token')
        
        # Assert log was called
        self.assertTrue(hasattr(diarizer, 'log'))
    
    @patch('src.audio_processing.diarization.Pipeline')
    def test_load_models(self, mock_pipeline):
        """
        Test loading diarization and VAD models
        """
        # Setup mock
        mock_pipeline.from_pretrained.return_value = MagicMock()
        
        # Create diarizer
        diarizer = SpeakerDiarizer(self.config)
        
        # Load models
        diarizer.load_models()
        
        # Assert models were loaded
        self.assertTrue(mock_pipeline.from_pretrained.called)
        self.assertEqual(mock_pipeline.from_pretrained.call_count, 3)  # Should load diarization, VAD, and segmentation models
        
        # Check model names
        calls = mock_pipeline.from_pretrained.call_args_list
        self.assertTrue(any('speaker-diarization' in str(call) for call in calls))
        self.assertTrue(any('voice-activity-detection' in str(call) for call in calls))
        self.assertTrue(any('segmentation' in str(call) for call in calls))
    
    @patch('src.audio_processing.diarization.Pipeline')
    def test_diarize_with_multiple_speaker_counts(self, mock_pipeline):
        """
        Test diarization with multiple speaker counts
        """
        # Setup mocks
        mock_pipeline.from_pretrained.return_value = MagicMock()
        mock_diarization = MagicMock()
        mock_segments = []
        
        # Create sample segments for 3 speakers
        for i in range(10):
            mock_segments.append(MagicMock(start=i, end=i+0.5, label=f"SPEAKER_{i%3}"))
        
        # Setup mock to return segments
        mock_diarization.itertracks.return_value = [(seg, None, seg.label) for seg in mock_segments]
        mock_pipeline.from_pretrained.return_value.return_value = mock_diarization
        
        # Create diarizer
        diarizer = SpeakerDiarizer(self.config)
        diarizer.diarization_pipeline = mock_pipeline.from_pretrained.return_value
        diarizer.vad_pipeline = mock_pipeline.from_pretrained.return_value
        
        # Run diarization
        result = diarizer.diarize(self.input_file, self.output_dir)
        
        # Assert diarization was called with different speaker counts
        self.assertTrue(result)
        
        # Check that the pipeline was called multiple times with different speaker counts
        calls = diarizer.diarization_pipeline.call_args_list
        self.assertGreaterEqual(len(calls), 3)  # Should try at least 3 speaker counts
    
    @patch('src.audio_processing.diarization.Pipeline')
    def test_diarize_with_vad(self, mock_pipeline):
        """
        Test diarization with Voice Activity Detection
        """
        # Setup mocks
        mock_pipeline.from_pretrained.return_value = MagicMock()
        
        # Mock VAD result
        mock_vad_result = MagicMock()
        mock_vad_segments = []
        
        # Create sample VAD segments
        for i in range(5):
            mock_vad_segments.append(MagicMock(start=i, end=i+0.8))
        
        # Setup mock to return VAD segments
        mock_vad_result.itertracks.return_value = [(seg, None, "SPEECH") for seg in mock_vad_segments]
        
        # Mock diarization result
        mock_diarization = MagicMock()
        mock_segments = []
        
        # Create sample segments for 3 speakers
        for i in range(10):
            mock_segments.append(MagicMock(start=i, end=i+0.5, label=f"SPEAKER_{i%3}"))
        
        # Setup mock to return segments
        mock_diarization.itertracks.return_value = [(seg, None, seg.label) for seg in mock_segments]
        
        # Setup pipeline returns
        mock_pipeline.from_pretrained.return_value.side_effect = [mock_vad_result, mock_diarization]
        
        # Create diarizer
        diarizer = SpeakerDiarizer(self.config)
        diarizer.diarization_pipeline = mock_pipeline.from_pretrained.return_value
        diarizer.vad_pipeline = mock_pipeline.from_pretrained.return_value
        
        # Run diarization
        result = diarizer.diarize(self.input_file, self.output_dir)
        
        # Assert diarization was successful
        self.assertTrue(result)
        
        # Check that VAD was used
        self.assertTrue(diarizer.vad_pipeline.called)
    
    @patch('src.audio_processing.diarization.Pipeline')
    def test_diarize_with_debug(self, mock_pipeline):
        """
        Test that debug files are created when debug mode is enabled
        """
        # Setup mocks
        mock_pipeline.from_pretrained.return_value = MagicMock()
        mock_diarization = MagicMock()
        mock_segments = []
        
        # Create sample segments
        for i in range(10):
            mock_segments.append(MagicMock(start=i, end=i+0.5, label=f"SPEAKER_{i%3}"))
        
        # Setup mock to return segments
        mock_diarization.itertracks.return_value = [(seg, None, seg.label) for seg in mock_segments]
        mock_pipeline.from_pretrained.return_value.return_value = mock_diarization
        
        # Create diarizer with debug enabled
        config = self.config.copy()
        config['debug'] = True
        diarizer = SpeakerDiarizer(config)
        diarizer.diarization_pipeline = mock_pipeline.from_pretrained.return_value
        diarizer.vad_pipeline = mock_pipeline.from_pretrained.return_value
        
        # Run diarization
        result = diarizer.diarize(self.input_file, self.output_dir)
        
        # Assert diarization was successful
        self.assertTrue(result)
        
        # Check that debug files were created
        debug_files = os.listdir(config['debug_dir'])
        self.assertGreater(len(debug_files), 0)
        
        # Check that there are segment files for each speaker count
        segment_files = [f for f in os.listdir(self.output_dir) if f.endswith('.segments')]
        self.assertGreater(len(segment_files), 0)


if __name__ == '__main__':
    unittest.main()
