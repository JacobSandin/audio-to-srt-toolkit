#!/usr/bin/env python3
# Test file for progress bar handling in the _run_demucs method
# Tests that subprocess output is properly handled
# 2025-04-23 -JS

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import pytest
import subprocess

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import modules to test
from src.audio_processing.preprocessor import AudioPreprocessor


class TestProgressBarHandling(unittest.TestCase):
    """
    Test case for progress bar handling in the _run_demucs method
    """
    
    def setUp(self):
        """
        Set up test environment before each test
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "test_input.wav")
        self.output_file = os.path.join(self.temp_dir.name, "test_output.wav")
        
        # Create a simple test audio file
        with open(self.input_file, 'wb') as f:
            f.write(b'test audio data')
        
        # Create preprocessor instance
        self.config = {}
        self.preprocessor = AudioPreprocessor(self.config)
    
    def tearDown(self):
        """
        Clean up after each test
        """
        self.temp_dir.cleanup()
    
    @patch('os.path.exists')
    @patch('pydub.AudioSegment.from_file')
    @patch('subprocess.run')
    def test_run_demucs_subprocess_handling(self, mock_run, mock_from_file, mock_exists):
        """
        Test that _run_demucs correctly handles subprocess output
        """
        # Configure mocks
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b'Separated tracks will be stored in /tmp/test_dir'
        mock_run.return_value = mock_process
        
        mock_exists.return_value = True
        mock_audio = MagicMock()
        mock_from_file.return_value = mock_audio
        
        # Call the method
        result = self.preprocessor._run_demucs(self.input_file, self.output_file)
        
        # Assertions
        self.assertTrue(mock_run.called)
        self.assertTrue(result)
    
    @patch('os.path.exists')
    @patch('pydub.AudioSegment.from_file')
    @patch('subprocess.run')
    def test_run_demucs_progress_output(self, mock_run, mock_from_file, mock_exists):
        """
        Test that _run_demucs correctly handles progress output from demucs
        """
        # Configure mocks
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b'Separated tracks will be stored in /tmp/test_dir'
        mock_run.return_value = mock_process
        
        mock_exists.return_value = True
        mock_audio = MagicMock()
        mock_from_file.return_value = mock_audio
        
        # Call the method
        result = self.preprocessor._run_demucs(self.input_file, self.output_file)
        
        # Assertions
        self.assertTrue(mock_run.called)
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_run_demucs_error_handling(self, mock_run):
        """
        Test that _run_demucs correctly handles errors from demucs
        """
        # Configure mock
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = b'Error: something went wrong'
        mock_run.return_value = mock_process
        
        # Call the method
        result = self.preprocessor._run_demucs(self.input_file, self.output_file)
        
        # Assertions
        self.assertTrue(mock_run.called)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
