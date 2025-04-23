#!/usr/bin/env python3
# Test file for SRT generator functionality
# Tests the creation of SRT files from diarization results
# 2025-04-23 - JS

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
from src.audio_processing.srt_generator import SRTGenerator


class TestSRTGenerator(unittest.TestCase):
    """
    Test case for SRT generator functionality
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock diarization result
        self.diarization_result = [
            {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01", "text": ""},
            {"start": 3.0, "end": 5.5, "speaker": "SPEAKER_02", "text": ""},
            {"start": 6.0, "end": 8.0, "speaker": "SPEAKER_01", "text": ""},
            {"start": 9.0, "end": 12.0, "speaker": "SPEAKER_02", "text": ""}
        ]
        
        # Create SRT generator instance
        self.srt_generator = SRTGenerator()
    
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    def test_format_timestamp(self):
        """
        Test that timestamps are formatted correctly for SRT files
        """
        # Test various timestamp formats
        test_cases = [
            (0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.25, "00:01:01,250"),
            (3661.75, "01:01:01,750")
        ]
        
        for seconds, expected in test_cases:
            result = self.srt_generator.format_timestamp(seconds)
            self.assertEqual(result, expected, f"Timestamp {seconds} should format to {expected}, got {result}")
    
    def test_generate_srt(self):
        """
        Test that SRT files are generated correctly from diarization results
        """
        # Generate SRT file
        output_file = os.path.join(self.output_dir, "test_output.srt")
        self.srt_generator.generate_srt(self.diarization_result, output_file)
        
        # Verify the SRT file was created
        self.assertTrue(os.path.exists(output_file), "SRT file was not created")
        
        # Read the content of the SRT file
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify the content has the correct format
        # SRT files have a specific format: index, timestamp, text, blank line
        lines = content.strip().split('\n')
        
        # Should have 4 entries × 3 lines each (index, timestamp, text) + 3 blank lines = 15 lines
        # The last blank line is removed by strip(), so we expect 15 lines
        self.assertEqual(len(lines), 15, f"Expected 15 lines in SRT file, got {len(lines)}")  # 2025-04-23 - JS
        
        # Check the first entry
        self.assertEqual(lines[0], "1", "First line should be the index '1'")
        self.assertEqual(lines[1], "00:00:00,500 --> 00:00:02,500", 
                         "Second line should be the timestamp range")
        self.assertEqual(lines[2], "SPEAKER_01:", "Third line should be the speaker label")
        
        # Check the second entry
        self.assertEqual(lines[4], "2", "Fifth line should be the index '2'")
        self.assertEqual(lines[5], "00:00:03,000 --> 00:00:05,500", 
                         "Sixth line should be the timestamp range")
        self.assertEqual(lines[6], "SPEAKER_02:", "Seventh line should be the speaker label")
    
    def test_generate_srt_with_text(self):
        """
        Test that SRT files are generated correctly with transcribed text
        """
        # Create a mock diarization result with text
        diarization_result_with_text = [
            {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01", "text": "Hello, how are you?"},
            {"start": 3.0, "end": 5.5, "speaker": "SPEAKER_02", "text": "I'm fine, thank you."},
            {"start": 6.0, "end": 8.0, "speaker": "SPEAKER_01", "text": "That's good to hear."},
            {"start": 9.0, "end": 12.0, "speaker": "SPEAKER_02", "text": "Yes, it's a nice day."}
        ]
        
        # Generate SRT file
        output_file = os.path.join(self.output_dir, "test_output_with_text.srt")
        self.srt_generator.generate_srt(diarization_result_with_text, output_file)
        
        # Verify the SRT file was created
        self.assertTrue(os.path.exists(output_file), "SRT file was not created")
        
        # Read the content of the SRT file
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify the content has the correct format with text
        lines = content.strip().split('\n')
        
        # Check that the text is included
        self.assertEqual(lines[2], "SPEAKER_01: Hello, how are you?", 
                         "Third line should include the speaker label and text")
        self.assertEqual(lines[6], "SPEAKER_02: I'm fine, thank you.", 
                         "Seventh line should include the speaker label and text")
    
    def test_generate_srt_with_custom_format(self):
        """
        Test that SRT files can be generated with custom formatting
        """
        # Generate SRT file with custom format
        output_file = os.path.join(self.output_dir, "test_output_custom.srt")
        self.srt_generator.generate_srt(
            self.diarization_result, 
            output_file,
            speaker_format="Speaker {speaker_id}:",
            include_timestamps=True
        )
        
        # Verify the SRT file was created
        self.assertTrue(os.path.exists(output_file), "SRT file was not created")
        
        # Read the content of the SRT file
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verify the content has the custom format
        lines = content.strip().split('\n')
        
        # Check the custom speaker format
        self.assertTrue(lines[2].startswith("Speaker 01:"), 
                        f"Expected line to start with 'Speaker 01:', got '{lines[2]}'")
        
        # Check that timestamps are included in the text
        self.assertTrue("[00:00:00,500 --> 00:00:02,500]" in lines[2], 
                        f"Expected timestamp in text, got '{lines[2]}'")


if __name__ == '__main__':
    unittest.main()
