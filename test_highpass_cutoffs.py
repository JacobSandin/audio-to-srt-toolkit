#!/usr/bin/env python3
# Test different highpass cutoff frequencies
# 2025-04-23 -JS
#
# Usage: ./test_highpass_cutoffs.py --input your_audio_file.mp3 --include-original --debug
#        ./test_highpass_cutoffs.py --input your_audio_file.mp3 --cutoffs 200 300 400 500

import os
import sys
import argparse
import logging
import tempfile
import datetime
from pathlib import Path

# Add the current directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the audio toolkit modules
from src.audio_processing.preprocessor import AudioPreprocessor

def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Test different highpass cutoff frequencies')
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input audio file to process'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Directory to save output files (default: ./output)'
    )
    
    parser.add_argument(
        '--cutoffs',
        type=int,
        nargs='+',
        default=[500, 1000,1500, 2000,2500,3000,3500,4000,4500,5000],
        help='List of highpass cutoff frequencies to test (default: 150 250 350 450)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode to save intermediate files'
    )
    
    parser.add_argument(
        '--include-original',
        action='store_true',
        help='Include a copy of the original audio file for comparison'
    )
    
    return parser.parse_args()

def main():
    """
    Main entry point for the script.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Ensure input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return False
    
    # Generate timestamp for this test run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get input filename without extension
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test run subdirectory with timestamp
    test_run_dir = os.path.join(args.output_dir, f"{timestamp}_{input_basename}")
    os.makedirs(test_run_dir, exist_ok=True)
    logger.info(f"Created test run directory: {test_run_dir}")
    
    # Create debug directory if debug mode is enabled
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(test_run_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        logger.info(f"Created debug directory: {debug_dir}")
    
    # Process the audio with different highpass cutoff frequencies
    logger.info(f"Testing highpass cutoff frequencies: {args.cutoffs}")
    
    # If requested, copy the original file for comparison
    if args.include_original:
        original_copy = os.path.join(test_run_dir, f"{input_basename}_original.wav")
        logger.info(f"Copying original file for comparison: {original_copy}")
        try:
            from pydub import AudioSegment
            original_audio = AudioSegment.from_file(args.input)
            original_audio.export(original_copy, format="wav")
            logger.info(f"Original file saved for comparison: {original_copy}")
        except Exception as e:
            logger.error(f"Error copying original file: {str(e)}")
    
    for cutoff in args.cutoffs:
        logger.info(f"Processing with highpass cutoff: {cutoff}Hz")
        
        # Create output file path for this cutoff
        output_file = os.path.join(test_run_dir, f"{input_basename}_highpass_{cutoff}hz.wav")
        
        # Create preprocessor with specific highpass cutoff
        config = {
            'debug': args.debug,
            'debug_dir': debug_dir,
            'highpass_cutoff': cutoff
        }
        preprocessor = AudioPreprocessor(config)
        
        # Process the audio with only highpass filtering
        try:
            # Load audio
            logger.info(f"Loading audio file: {args.input}")
            from pydub import AudioSegment
            audio = AudioSegment.from_file(args.input)
            sample_rate = audio.frame_rate
            logger.info(f"Audio loaded: {len(audio)/1000:.2f} seconds, {sample_rate}Hz sample rate")
            
            # Apply only highpass filter
            logger.info(f"Applying high-pass filter (cutoff: {cutoff}Hz)...")
            filtered_audio = preprocessor.apply_highpass(audio, cutoff=cutoff, sample_rate=sample_rate)
            
            # Save the filtered audio
            filtered_audio.export(output_file, format="wav")
            logger.info(f"Successfully processed with highpass cutoff: {cutoff}Hz")
            logger.info(f"Output file: {output_file}")
            
            # Save debug file if debug mode is enabled
            if args.debug and debug_dir:
                debug_file = os.path.join(debug_dir, f"{input_basename}_highpass_{cutoff}hz_debug.wav")
                filtered_audio.export(debug_file, format="wav")
                logger.info(f"Debug file saved: {debug_file}")
        
        except Exception as e:
            logger.error(f"Error processing with highpass cutoff {cutoff}Hz: {str(e)}")
            logger.error(f"Stack trace: ", exc_info=True)
    
    # Print summary of results
    logger.info("=" * 60)
    logger.info(f"Processing complete! All test files are in: {test_run_dir}")
    logger.info(f"Tested highpass cutoffs: {args.cutoffs}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"To compare results, listen to the output files in: {test_run_dir}")
    logger.info("=" * 60)
    return True

if __name__ == "__main__":
    main()
