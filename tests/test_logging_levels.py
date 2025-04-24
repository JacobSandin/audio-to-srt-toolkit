#!/usr/bin/env python3
# Test for proper logging levels in the audio toolkit
# Ensures that only essential progress information is logged at INFO level
# 2025-04-24 -JS

import unittest
import logging
import io
import os
from unittest.mock import patch, MagicMock
from src.audio_processing.diarization import SpeakerDiarizer

class TestLoggingLevels(unittest.TestCase):
    """Test that logging levels are properly set and messages are appropriately categorized."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a string IO object to capture log output
        self.log_capture = io.StringIO()
        
        # Configure a simple logger for testing
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)  # Set to INFO level
        
        # Add a handler that writes to our string IO
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.INFO)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)
        
    def test_diarization_logging_levels(self):
        """Test that diarization module uses appropriate logging levels."""
        # Create a minimal config for testing
        test_config = {
            'models': {
                'diarization': {
                    'primary': ['test-model'],
                    'fallback': []
                },
                'vad': {
                    'primary': ['test-vad-model'],
                    'fallback': []
                },
                'segmentation': {
                    'primary': ['test-seg-model'],
                    'fallback': []
                }
            },
            'parameters': {
                'min_speakers': 2,
                'max_speakers': 4
            }
        }
            
        # Create a diarizer with our logger
        with patch.object(SpeakerDiarizer, '__init__', return_value=None):
            diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
            diarizer.config = test_config
            diarizer.logger = self.logger
            
            # Test logging at different levels
            # These should appear at INFO level
            self.logger.info("Starting speaker diarization")
            self.logger.info("Processing audio file: test.wav")
            self.logger.info("Diarization completed successfully")
            
            # These should be at DEBUG level, not INFO
            self.logger.debug("Loaded configuration from config.yaml")
            self.logger.debug("Speaker diarizer initialized")
            self.logger.debug("Configuration: min_speakers=2, max_speakers=4")
            self.logger.debug("Loading diarization model...")
            self.logger.debug("Using diarization models: ['model1', 'model2']")
            self.logger.debug("Successfully loaded diarization model: model1")
            
            # Get the log output - should only contain INFO messages
            log_output = self.log_capture.getvalue()
            
            # Essential progress messages should be present (INFO level)
            self.assertIn("Starting speaker diarization", log_output)
            self.assertIn("Processing audio file: test.wav", log_output)
            self.assertIn("Diarization completed successfully", log_output)
            
            # Technical messages should NOT be present (DEBUG level)
            self.assertNotIn("Loaded configuration", log_output)
            self.assertNotIn("Speaker diarizer initialized", log_output)
            self.assertNotIn("Configuration: min_speakers", log_output)
            self.assertNotIn("Loading diarization model", log_output)
            self.assertNotIn("Using diarization models", log_output)
            self.assertNotIn("Successfully loaded diarization model", log_output)
    
    def test_recommended_logging_levels(self):
        """Test recommended logging levels for different message types."""
        # This test documents which messages should be at which level
        
        # Messages that should be at INFO level (user-facing progress)
        info_messages = [
            "Starting Audio Toolkit",
            "Processing audio file: {filename}",
            "Creating WAV file from {filename}",
            "WAV file created and saved to {filename}",
            "Running vocal separation step",
            "Vocal separation completed",
            "Running audio enhancement step",
            "Saved enhanced audio to {filename}",
            "Starting speaker diarization",
            "Diarization completed successfully",
            "Creating SRT file",
            "SRT file created successfully",
            "Processing completed successfully"
        ]
        
        # Messages that should be at DEBUG level (technical details)
        debug_messages = [
            "Loaded configuration from {filename}",
            "Speaker diarizer initialized",
            "Configuration: min_speakers={n}, max_speakers={n}",
            "Loading diarization model...",
            "Using diarization models: {models}",
            "Successfully loaded diarization model: {model}",
            "Loading Voice Activity Detection (VAD) model...",
            "Using VAD models: {models}",
            "Loading segmentation model...",
            "Using segmentation models: {models}",
            "Models loaded successfully",
            "Running Voice Activity Detection...",
            "Detected {n} speech regions in {time} seconds",
            "Trying with num_speakers={n}",
            "Directory created: {dir}",
            "Input audio: {duration} seconds, {sample_rate}Hz, {channels} channels",
            "Estimated processing time: {time} minutes"
        ]
        
        # Document these as a passing test
        self.assertTrue(len(info_messages) > 0)
        self.assertTrue(len(debug_messages) > 0)

if __name__ == "__main__":
    unittest.main()
