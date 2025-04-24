#!/usr/bin/env python3
# Test version compatibility layer for handling version differences
# between PyAnnote/PyTorch models and current library versions
# 2025-04-24 -JS

import unittest
import os
import sys
import tempfile
import logging
from unittest.mock import patch, MagicMock, PropertyMock
import torch
import importlib

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the module to be tested
from src.audio_processing.diarization import VersionCompatibilityLayer


class TestVersionCompatibilityLayer(unittest.TestCase):
    """
    Test the version compatibility layer that handles differences between
    model versions and current library versions.
    """

    def setUp(self):
        """Set up test fixtures."""
        # Configure logging to capture log messages
        self.log_capture = []
        
        def mock_log(level, message):
            self.log_capture.append((level, message))
        
        # Create a patcher for the logging
        self.log_patcher = patch('logging.log', mock_log)
        self.log_patcher.start()
        
        # Create an instance of the compatibility layer
        self.compatibility_layer = VersionCompatibilityLayer()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the logging patcher
        self.log_patcher.stop()
        
        # Clear the log capture
        self.log_capture = []

    def test_initialization(self):
        """Test that the compatibility layer initializes correctly."""
        # Verify that the compatibility layer has the expected attributes
        self.assertIsNotNone(self.compatibility_layer.current_pyannote_version)
        self.assertIsNotNone(self.compatibility_layer.current_torch_version)
        self.assertEqual(self.compatibility_layer.patched_modules, {})

    def test_get_package_version(self):
        """Test the _get_package_version method."""
        # Test with a package that should exist
        torch_version = self.compatibility_layer._get_package_version('torch')
        self.assertIsNotNone(torch_version)
        
        # Test with a package that should not exist
        nonexistent_version = self.compatibility_layer._get_package_version('nonexistent_package_12345')
        self.assertIsNone(nonexistent_version)

    @patch('torch.load')
    def test_patch_torch_for_older_models(self, mock_torch_load):
        """Test that PyTorch patching works correctly."""
        # Store the original torch.load function
        original_torch_load = torch.load
        
        # Call the patching method
        self.compatibility_layer._patch_torch_for_older_models()
        
        # Verify that torch.load has been patched
        self.assertNotEqual(torch.load, original_torch_load)
        self.assertIn('torch.load', self.compatibility_layer.patched_modules)
        
        # Call the patched function and verify it works as expected
        dummy_path = 'dummy_path'
        torch.load(dummy_path)
        
        # Verify that mock_torch_load was called with the expected arguments
        mock_torch_load.assert_called_once()
        args, kwargs = mock_torch_load.call_args
        self.assertEqual(args[0], dummy_path)
        self.assertIn('map_location', kwargs)

    def test_context_manager(self):
        """Test that the context manager applies and removes patches correctly."""
        # Create a model_info dictionary
        model_info = {
            'pyannote_version': '0.0.1',
            'torch_version': '1.7.1'
        }
        
        # Store the original torch.load function
        original_torch_load = torch.load
        
        # Use the context manager
        with self.compatibility_layer.patch_model_loading(model_info):
            # Verify that patches have been applied
            self.assertNotEqual(torch.load, original_torch_load)
        
        # Verify that patches have been removed after exiting the context
        self.assertEqual(torch.load, original_torch_load)
        self.assertEqual(self.compatibility_layer.patched_modules, {})

    @patch('importlib.import_module')
    def test_remove_patches(self, mock_import_module):
        """Test that patches are properly removed."""
        # Create a dummy module and function to patch
        dummy_module = MagicMock()
        dummy_function = MagicMock()
        
        # Add it to the patched_modules dictionary
        self.compatibility_layer.patched_modules = {
            'dummy.module.function': dummy_function
        }
        
        # Set up the mock to return our dummy module
        mock_import_module.return_value = dummy_module
        
        # Call the remove_patches method
        self.compatibility_layer._remove_patches()
        
        # Verify that importlib.import_module was called with the expected arguments
        mock_import_module.assert_called_once_with('dummy.module')
        
        # Verify that the patched_modules dictionary is empty
        self.assertEqual(self.compatibility_layer.patched_modules, {})

    def test_version_comparison(self):
        """Test that version comparison works correctly."""
        # Create a model_info dictionary with older versions
        model_info = {
            'pyannote_version': '0.0.1',
            'torch_version': '1.7.1'
        }
        
        # Directly patch the version comparison logic instead of the properties
        with patch('src.audio_processing.diarization.version.parse') as mock_version_parse:
            # Configure the mock to return appropriate version objects
            def side_effect(version_str):
                mock_version = MagicMock()
                if version_str == '0.0.1':
                    # PyAnnote model version
                    mock_version.major = 0
                elif version_str == '1.7.1':
                    # PyTorch model version
                    mock_version.major = 1
                elif version_str in ('0.0.0', None):
                    # Default value
                    mock_version.major = 0
                else:
                    # Current versions
                    if '3' in version_str:
                        # PyAnnote current version
                        mock_version.major = 3
                    elif '2' in version_str:
                        # PyTorch current version
                        mock_version.major = 2
                return mock_version
            
            mock_version_parse.side_effect = side_effect
            
            # Set current versions for the test
            self.compatibility_layer.current_pyannote_version = '3.1.1'
            self.compatibility_layer.current_torch_version = '2.6.0'
            
            # Mock the patch methods to verify they're called
            with patch.object(self.compatibility_layer, '_patch_pyannote_for_older_models') as mock_patch_pyannote:
                with patch.object(self.compatibility_layer, '_patch_torch_for_older_models') as mock_patch_torch:
                    # Apply the patches
                    self.compatibility_layer._apply_patches(model_info)
                    
                    # Verify that both patch methods were called
                    mock_patch_pyannote.assert_called_once()
                    mock_patch_torch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
