#!/usr/bin/env python3
# Test for continuation feature in the audio toolkit
# Ensures that processing can be resumed from a specific step
# 2025-04-24 -JS

import unittest
import os
import tempfile
import shutil
import json
import sys
import logging
from pathlib import Path

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_toolkit import parse_args
from src.audio_processing.diarization import SpeakerDiarizer

class TestContinuationFeature(unittest.TestCase):
    """Test the core functionality of the continuation feature."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock segments file
        self.segments_file = os.path.join(self.test_dir, "test_audio.3speakers.segments")
        self.mock_segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01", "text": ""},
            {"start": 1.5, "end": 2.5, "speaker": "SPEAKER_02", "text": ""},
            {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_03", "text": ""}
        ]
        with open(self.segments_file, "w") as f:
            json.dump(self.mock_segments, f)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_load_segments(self):
        """Test loading segments from a file."""
        # Create a diarizer instance
        diarizer = SpeakerDiarizer()
        
        # Load segments
        loaded_segments = diarizer.load_segments(self.segments_file)
        
        # Verify that segments were loaded correctly
        self.assertEqual(len(loaded_segments), 3)
        self.assertEqual(loaded_segments[0]["speaker"], "SPEAKER_01")
        self.assertEqual(loaded_segments[1]["speaker"], "SPEAKER_02")
        self.assertEqual(loaded_segments[2]["speaker"], "SPEAKER_03")
        
        # Verify that diarization_segments attribute was set
        self.assertTrue(hasattr(diarizer, 'diarization_segments'))
        self.assertEqual(diarizer.diarization_segments, loaded_segments)
    
    def test_continuation_args(self):
        """Test that continuation arguments are parsed correctly."""
        # Test with continue_from=diarization
        # Instead of modifying sys.argv directly, pass arguments to parse_args
        # Note that input-audio and continue-folder are mutually exclusive
        # Don't include the script name in the arguments list
        # 2025-04-24 -JS
        test_args = ['--continue-from', 'diarization', 
                    '--continue-folder', '/path/to/output',
                    '--speaker-count', '3']
        args = parse_args(test_args)
        
        self.assertEqual(args.continue_from, "diarization")
        self.assertEqual(args.continue_folder, "/path/to/output")
        self.assertEqual(args.speaker_count, 3)
        
        # Test with continue_from=srt
        # Don't include the script name in the arguments list
        # 2025-04-24 -JS
        test_args = ['--continue-from', 'srt', 
                    '--continue-folder', '/path/to/output']
        args = parse_args(test_args)
        
        self.assertEqual(args.continue_from, "srt")
        self.assertEqual(args.continue_folder, "/path/to/output")

if __name__ == "__main__":
    unittest.main()
