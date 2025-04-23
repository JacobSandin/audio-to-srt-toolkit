import unittest
from unittest.mock import patch, MagicMock
import logging
import sys
import os
import io

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_toolkit

class TestDebugFilesOnly(unittest.TestCase):
    """Test the --debug-files-only switch functionality."""
    
    @patch('logging.FileHandler')
    @patch('logging.StreamHandler')
    @patch('logging.NullHandler')
    @patch('logging.basicConfig')
    def test_debug_files_only_flag(self, mock_basicConfig, mock_NullHandler, 
                                  mock_StreamHandler, mock_FileHandler):
        """Test that --debug-files-only enables debug logging to file but not console."""
        # Create mock args with debug_files_only=True
        args = MagicMock()
        args.debug = False
        args.quiet = False
        args.debug_files_only = True
        
        # Mock file handler
        mock_file_handler_instance = MagicMock()
        mock_FileHandler.return_value = mock_file_handler_instance
        
        # Mock null handler
        mock_null_handler_instance = MagicMock()
        mock_NullHandler.return_value = mock_null_handler_instance
        
        # Call setup_logging
        audio_toolkit.setup_logging(args)
        
        # Verify that basicConfig was called with DEBUG level
        mock_basicConfig.assert_called_once()
        call_args = mock_basicConfig.call_args[1]
        self.assertEqual(call_args['level'], logging.DEBUG)
        
        # Verify that FileHandler was created
        mock_FileHandler.assert_called_once()
        
        # Verify that NullHandler was used for console (not StreamHandler)
        mock_StreamHandler.assert_not_called()
        mock_NullHandler.assert_called_once()
        
        # Verify handlers were passed to basicConfig
        self.assertEqual(len(call_args['handlers']), 2)
        self.assertIn(mock_file_handler_instance, call_args['handlers'])
        self.assertIn(mock_null_handler_instance, call_args['handlers'])

if __name__ == '__main__':
    unittest.main()
