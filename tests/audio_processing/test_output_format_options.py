#!/usr/bin/env python3
# Test output format options for audio processing
# 2025-04-24 -JS

import unittest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the module to be tested
from src.audio_processing.preprocessor import AudioPreprocessor


class TestOutputFormatOptions(unittest.TestCase):
    """
    Test the output format options (MP3 only, WAV only, or both) for audio processing.
    """

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a basic configuration
        self.config = {
            'debug': False,
            'debug_dir': None,
            'bit_depth': 16,
            'sample_rate': 44100,
            'highpass_cutoff': 300,
            'lowpass_cutoff': 8000,
            'compression_threshold': -10.0,
            'compression_ratio': 2.0,
            'volume_gain': 12.0,
            'skip_steps': []
        }

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    @patch('src.audio_processing.preprocessor.AudioSegment')
    def test_wav_only_output(self, mock_audio_segment):
        """Test that only WAV files are created when output_format is set to 'wav'."""
        # Configure the mock
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.export.return_value = MagicMock()
        
        # Add output_format to config
        self.config['output_format'] = 'wav'
        
        # Create an instance of AudioPreprocessor
        preprocessor = AudioPreprocessor(self.config)
        
        # Create test input and output files
        input_file = os.path.join(self.temp_dir.name, 'input.mp3')
        output_file = os.path.join(self.temp_dir.name, 'output.wav')
        
        # Create a dummy input file
        with open(input_file, 'w') as f:
            f.write('dummy content')
        
        # Process the audio file
        preprocessor.process_audio(input_file, output_file)
        
        # Check that export was called with the correct format
        export_calls = mock_audio.export.call_args_list
        
        # There should be calls to export with format='wav' but none with format='mp3'
        wav_calls = [call for call in export_calls if 'format' in call[1] and call[1]['format'] == 'wav']
        mp3_calls = [call for call in export_calls if 'format' in call[1] and call[1]['format'] == 'mp3']
        
        self.assertTrue(len(wav_calls) > 0, "No calls to export with format='wav'")
        self.assertEqual(len(mp3_calls), 0, "Found calls to export with format='mp3' when output_format='wav'")

    @patch('src.audio_processing.preprocessor.AudioSegment')
    def test_mp3_only_output(self, mock_audio_segment):
        """Test that only MP3 files are created when output_format is set to 'mp3'."""
        # Configure the mock
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.export.return_value = MagicMock()
        
        # Add output_format to config
        self.config['output_format'] = 'mp3'
        
        # Create an instance of AudioPreprocessor
        preprocessor = AudioPreprocessor(self.config)
        
        # Create test input and output files
        input_file = os.path.join(self.temp_dir.name, 'input.wav')
        output_file = os.path.join(self.temp_dir.name, 'output.mp3')
        
        # Create a dummy input file
        with open(input_file, 'w') as f:
            f.write('dummy content')
        
        # Process the audio file
        preprocessor.process_audio(input_file, output_file)
        
        # Check that export was called with the correct format
        export_calls = mock_audio.export.call_args_list
        
        # There should be calls to export with format='mp3' but none with format='wav'
        # (except for the final output which should match the output_file extension)
        mp3_calls = [call for call in export_calls if 'format' in call[1] and call[1]['format'] == 'mp3']
        wav_calls = [call for call in export_calls if 'format' in call[1] and call[1]['format'] == 'wav']
        
        # The final output should match the output_file extension
        self.assertTrue(len(mp3_calls) > 0, "No calls to export with format='mp3'")
        
        # There should be no WAV files created for intermediate steps
        intermediate_wav_calls = [call for call in wav_calls if 'debug' in str(call)]
        self.assertEqual(len(intermediate_wav_calls), 0, "Found calls to export intermediate files with format='wav' when output_format='mp3'")

    @patch('src.audio_processing.preprocessor.AudioSegment')
    def test_both_output_formats(self, mock_audio_segment):
        """Test that both WAV and MP3 files are created when output_format is set to 'both'."""
        # Configure the mock
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.export.return_value = MagicMock()
        
        # Add output_format to config
        self.config['output_format'] = 'both'
        
        # Create an instance of AudioPreprocessor
        preprocessor = AudioPreprocessor(self.config)
        
        # Create test input and output files
        input_file = os.path.join(self.temp_dir.name, 'input.wav')
        output_file = os.path.join(self.temp_dir.name, 'output.wav')
        
        # Create a dummy input file
        with open(input_file, 'w') as f:
            f.write('dummy content')
        
        # Process the audio file
        preprocessor.process_audio(input_file, output_file)
        
        # Check that export was called with both formats
        export_calls = mock_audio.export.call_args_list
        
        # There should be calls to export with both format='wav' and format='mp3'
        wav_calls = [call for call in export_calls if 'format' in call[1] and call[1]['format'] == 'wav']
        mp3_calls = [call for call in export_calls if 'format' in call[1] and call[1]['format'] == 'mp3']
        
        self.assertTrue(len(wav_calls) > 0, "No calls to export with format='wav'")
        self.assertTrue(len(mp3_calls) > 0, "No calls to export with format='mp3'")


if __name__ == '__main__':
    unittest.main()
