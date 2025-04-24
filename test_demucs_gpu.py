#!/usr/bin/env python3
# test_demucs_gpu.py - Test script for Demucs GPU acceleration
# 2025-04-24 -JS

import os
import sys
import time
import logging
import argparse
import subprocess
import torch
from pydub import AudioSegment

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import modules to test
from src.audio_processing.preprocessor import AudioPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_demucs_gpu')

def test_demucs_gpu(input_file, output_dir=None):
    """
    Test Demucs GPU acceleration with the given input file
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output files (default: current directory)
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return False
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create output file path
    basename = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"{basename}.vocals.{timestamp}.wav")
    
    # Print GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Get input file info
    try:
        input_audio = AudioSegment.from_file(input_file)
        logger.info(f"Input audio: {input_audio.duration_seconds:.2f} seconds, {input_audio.frame_rate}Hz, {input_audio.channels} channels")
        logger.info(f"Estimated processing time: {input_audio.duration_seconds / 60:.1f} minutes (varies by CPU/GPU speed)")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return False
    
    # Create preprocessor instance
    config = {}
    preprocessor = AudioPreprocessor(config)
    
    # Start timer
    start_time = time.time()
    
    # Run Demucs
    logger.info(f"Running Demucs on {input_file}")
    result = preprocessor._run_demucs(input_file, output_file)
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if result:
        logger.info(f"Demucs completed successfully in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        logger.info(f"Output file: {output_file}")
        return True
    else:
        logger.error(f"Demucs failed after {elapsed_time:.2f} seconds")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Demucs GPU acceleration")
    parser.add_argument("--input", "-i", default="Cardo recording 1.mp3", help="Input audio file")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    args = parser.parse_args()
    
    test_demucs_gpu(args.input, args.output_dir)
