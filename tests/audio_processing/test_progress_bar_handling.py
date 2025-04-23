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
    
    def test_run_demucs_subprocess_handling(self):
        """
        Test that _run_demucs correctly handles subprocess output
        """
        # Create a mock CompletedProcess with simulated output
        mock_process = MagicMock()
        mock_process.returncode = 0
        
        # Patch subprocess.run to return our mock process
        with patch('subprocess.run', return_value=mock_process) as mock_run:
            # Patch os.path.exists to make the method think the output file exists
            with patch('os.path.exists', return_value=True):
                # Patch shutil.copy to avoid actually copying files
                with patch('shutil.copy'):
                    # Call the _run_demucs method
                    result = self.preprocessor._run_demucs(self.input_file, self.output_file)
                    
                    # Verify that subprocess.run was called with the correct arguments
                    self.assertTrue(mock_run.called)
                    
                    # Check that the command includes demucs
                    args, kwargs = mock_run.call_args
                    cmd = args[0]
                    self.assertIn('demucs', cmd[0])
                    
                    # Verify that the method returned True (success)
                    self.assertTrue(result)
    
    def test_run_demucs_progress_output(self):
        """
        Test that _run_demucs correctly handles progress output from demucs
        """
        # Create a mock subprocess.run that simulates progress output
        def mock_subprocess_run(*args, **kwargs):
            # Simulate demucs progress output
            mock_process = MagicMock()
            mock_process.returncode = 0
            return mock_process
        
        # Patch subprocess.run with our mock function
        with patch('subprocess.run', side_effect=mock_subprocess_run) as mock_run:
            # Patch os.path.exists to make the method think the output file exists
            with patch('os.path.exists', return_value=True):
                # Patch shutil.copy to avoid actually copying files
                with patch('shutil.copy'):
                    # Call the _run_demucs method
                    result = self.preprocessor._run_demucs(self.input_file, self.output_file)
                    
                    # Verify that subprocess.run was called with the correct arguments
                    self.assertTrue(mock_run.called)
                    
                    # Check that stdout and stderr are not redirected (to allow tqdm to work)
                    args, kwargs = mock_run.call_args
                    self.assertNotIn('stdout', kwargs)
                    self.assertNotIn('stderr', kwargs)
                    
                    # Verify that the method returned True (success)
                    self.assertTrue(result)
    
    def test_run_demucs_error_handling(self):
        """
        Test that _run_demucs correctly handles errors from demucs
        """
        # Create a mock CompletedProcess with an error
        mock_process = MagicMock()
        mock_process.returncode = 1
        
        # Patch subprocess.run to return our mock process
        with patch('subprocess.run', return_value=mock_process) as mock_run:
            # Call the _run_demucs method
            result = self.preprocessor._run_demucs(self.input_file, self.output_file)
            
            # Verify that subprocess.run was called
            self.assertTrue(mock_run.called)
            
            # Verify that the method returned False (failure)
            self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
