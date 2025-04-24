#!/usr/bin/env python3
# Test for proper warning filtering in the audio toolkit
# Ensures that known dependency warnings are properly suppressed
# 2025-04-24 -JS

import unittest
import warnings
import io
import sys
from unittest.mock import patch, MagicMock
import pytest

class TestWarningFiltering(unittest.TestCase):
    """Test that known dependency warnings are properly filtered."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original filters
        self.original_filters = warnings.filters.copy()
        
        # Create a string buffer to capture warning output
        self.warning_output = io.StringIO()
        self.original_showwarning = warnings.showwarning
        
        # Define a function to capture warnings
        def capture_warning(message, category, filename, lineno, file=None, line=None):
            self.warning_output.write(str(message) + '\n')
        
        # Replace the showwarning function
        warnings.showwarning = capture_warning
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original warning filters and function
        warnings.filters = self.original_filters.copy()
        warnings.showwarning = self.original_showwarning
    
    def test_warning_filtering(self):
        """Test that known dependency warnings are filtered while others are not."""
        # Import the module that sets up the warning filters
        from audio_toolkit import setup_warning_filters
        
        # Call the function to set up filters
        setup_warning_filters()
        
        # Clear any previous warnings
        self.warning_output.seek(0)
        self.warning_output.truncate()
        
        # Generate warnings that should be filtered
        warnings.warn("torchaudio._backend.set_audio_backend has been deprecated", DeprecationWarning)
        warnings.warn("torchaudio._backend.get_audio_backend has been deprecated", DeprecationWarning)
        warnings.warn("`torchaudio.backend.common.AudioMetaData` has been moved to `torchaudio.AudioMetaData`", UserWarning)
        warnings.warn("Module 'speechbrain.pretrained' was deprecated", UserWarning)
        warnings.warn("NumExpr detected 8 cores", UserWarning)
        warnings.warn("TensorFloat-32 (TF32) has been disabled", UserWarning)
        warnings.warn("'audioop' is deprecated and slated for removal", DeprecationWarning)
        
        # Generate a warning that should not be filtered
        warnings.warn("This is an unrelated warning", UserWarning)
        
        # Get the captured warnings
        captured_warnings = self.warning_output.getvalue()
        
        # Check that filtered warnings don't appear
        self.assertNotIn("torchaudio._backend.set_audio_backend", captured_warnings)
        self.assertNotIn("torchaudio._backend.get_audio_backend", captured_warnings)
        self.assertNotIn("torchaudio.backend.common.AudioMetaData", captured_warnings)
        self.assertNotIn("speechbrain.pretrained", captured_warnings)
        self.assertNotIn("NumExpr detected", captured_warnings)
        self.assertNotIn("TensorFloat-32", captured_warnings)
        self.assertNotIn("audioop", captured_warnings)
        
        # Check that unfiltered warnings do appear
        self.assertIn("This is an unrelated warning", captured_warnings)
    


if __name__ == "__main__":
    unittest.main()
