#!/usr/bin/env python3
# Audio Toolkit - Main command-line interface
# Handles audio preprocessing, diarization, and SRT creation
# 2025-04-23 -JS

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.audio_processing.preprocessor import AudioPreprocessor


def log(level, *messages, **kwargs):
    """
    Unified logging function.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        messages: Messages to log
        kwargs: Additional logging parameters
    """
    logger = logging.getLogger(__name__)
    
    if level == logging.DEBUG:
        logger.debug(*messages, **kwargs)
    elif level == logging.INFO:
        logger.info(*messages, **kwargs)
    elif level == logging.WARNING:
        logger.warning(*messages, **kwargs)
    elif level == logging.ERROR:
        logger.error(*messages, **kwargs)
    elif level == logging.CRITICAL:
        logger.critical(*messages, **kwargs)


def setup_logging(args):
    """
    Set up logging based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(logs_dir, f'audio-toolkit-{timestamp}.log')
    
    # Determine log level
    if args.debug:
        console_level = logging.DEBUG
        file_level = logging.DEBUG
    elif args.quiet:
        console_level = logging.WARNING
        file_level = logging.INFO
    else:
        console_level = logging.INFO
        file_level = logging.DEBUG
    
    # Configure logging
    logging.basicConfig(
        level=min(console_level, file_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if not args.quiet or args.debug else logging.NullHandler()
        ]
    )
    
    log(logging.INFO, f"Logging initialized. Log file: {log_file}")


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Audio Toolkit - Process, diarize, and create SRT files from audio recordings'
    )
    
    parser.add_argument(
        '--input-audio',
        required=True,
        help='Path to input audio file'
    )
    
    parser.add_argument(
        '--output-dir',
        default=os.getcwd(),
        help='Directory to save output files (default: current directory)'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip audio preprocessing steps'
    )
    
    # WAV conversion parameters
    parser.add_argument(
        '--bit-depth',
        type=int,
        default=24,
        choices=[16, 24, 32],
        help='Bit depth for WAV conversion (default: 24)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        choices=[44100, 48000, 96000],
        help='Sample rate in Hz for WAV conversion (default: 48000)'
    )
    
    # Audio processing parameters
    parser.add_argument(
        '--highpass',
        type=int,
        default=150,
        help='High-pass filter cutoff frequency in Hz (default: 150)'
    )
    
    parser.add_argument(
        '--lowpass',
        type=int,
        default=8000,
        help='Low-pass filter cutoff frequency in Hz (default: 8000)'
    )
    
    parser.add_argument(
        '--compression-threshold',
        type=float,
        default=-10.0,
        help='Compression threshold in dB (default: -10.0)'
    )
    
    parser.add_argument(
        '--compression-ratio',
        type=float,
        default=2.0,
        help='Compression ratio (default: 2.0)'
    )
    
    parser.add_argument(
        '--volume-gain',
        type=float,
        default=6.0,
        help='Volume gain in dB (default: 6.0)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def process_audio(args):
    """
    Process audio file according to command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Get input file path
    input_file = os.path.abspath(args.input_audio)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file name based on input file
    input_basename = os.path.basename(input_file)
    output_basename = os.path.splitext(input_basename)[0] + "_processed.mp3"
    output_file = os.path.join(output_dir, output_basename)
    
    # Create debug directory if debug mode is enabled
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        log(logging.INFO, f"Debug mode enabled, intermediate files will be saved to {debug_dir}")
    
    # Configure audio preprocessor
    config = {
        'debug': args.debug,
        'debug_dir': debug_dir,
        # WAV conversion parameters
        'bit_depth': args.bit_depth,
        'sample_rate': args.sample_rate,
        # Audio processing parameters
        'highpass_cutoff': args.highpass,
        'lowpass_cutoff': args.lowpass,
        'compression_threshold': args.compression_threshold,
        'compression_ratio': args.compression_ratio,
        'volume_gain': args.volume_gain
    }
    
    log(logging.INFO, f"Processing audio file: {input_file}")
    log(logging.INFO, f"Output will be saved to: {output_file}")
    log(logging.INFO, f"Using {config['bit_depth']}-bit depth and {config['sample_rate']}Hz sample rate")
    
    # Import here to ensure the mock in tests works correctly
    from src.audio_processing.preprocessor import AudioPreprocessor
    preprocessor = AudioPreprocessor(config)
    
    # Preprocess audio
    if args.skip_preprocessing:
        log(logging.INFO, "Skipping preprocessing as requested")
        return True
    else:
        return preprocessor.preprocess(input_file, output_file)


def main():
    """
    Main entry point for the audio toolkit.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args)
    
    try:
        log(logging.INFO, "Starting Audio Toolkit")
        
        # Process audio
        if process_audio(args):
            log(logging.INFO, "Audio processing completed successfully")
            sys.exit(0)
        else:
            log(logging.ERROR, "Audio processing failed")
            sys.exit(1)
            
    except Exception as e:
        log(logging.CRITICAL, f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
