#!/usr/bin/env python3
# test_demucs_gpu.py - Test script for Demucs GPU acceleration
# 2025-04-24 -JS

import os
import sys
import time
import logging
import argparse
import subprocess
import threading
import torch
from pydub import AudioSegment

# Try to import GPU monitoring libraries
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("pynvml not available. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nvidia-ml-py"])
        import pynvml
        PYNVML_AVAILABLE = True
        print("pynvml installed successfully")
    except Exception as e:
        print(f"Failed to install pynvml: {e}")
        PYNVML_AVAILABLE = False

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

# Global variables for GPU monitoring
gpu_utilization = []
gpu_memory_used = []
monitoring_active = False

def monitor_gpu():
    """
    Monitor GPU utilization and memory usage
    """
    global gpu_utilization, gpu_memory_used, monitoring_active
    
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return
    
    try:
        # Initialize NVML
        pynvml.nvmlInit()
        
        # Get handle for GPU 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Monitor GPU while active
        while monitoring_active:
            try:
                # Get GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Get GPU memory usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / (1024 * 1024)  # Convert to MB
                
                # Store values
                gpu_utilization.append(gpu_util)
                gpu_memory_used.append(mem_used)
                
                # Print current values
                print(f"\rGPU Utilization: {gpu_util}%, Memory Used: {mem_used:.1f} MB", end="")
                
                # Sleep for a short time
                time.sleep(1)
            except Exception as e:
                print(f"\nError monitoring GPU: {e}")
                break
        
        # Shutdown NVML
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"\nError initializing GPU monitoring: {e}")

def test_demucs_gpu(input_file, output_dir=None, max_duration=None):
    """
    Test Demucs GPU acceleration with the given input file
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output files (default: current directory)
        max_duration: Maximum duration in seconds to process (default: process entire file)
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
        duration_sec = input_audio.duration_seconds
        
        # Apply max_duration if specified
        if max_duration and max_duration > 0 and max_duration < duration_sec:
            logger.info(f"Limiting test to {max_duration} seconds of the {duration_sec:.2f} second file")
            
            # Create a temporary file with the limited duration
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, f"temp_{os.path.basename(input_file)}")
            
            # Extract the segment and export to temp file
            segment = input_audio[:max_duration * 1000]  # Convert to milliseconds
            segment.export(temp_file, format=os.path.splitext(input_file)[1][1:])  # Use same format as input
            
            # Update input file and duration
            input_file = temp_file
            duration_sec = max_duration
            input_audio = segment
        
        logger.info(f"Input audio: {duration_sec:.2f} seconds, {input_audio.frame_rate}Hz, {input_audio.channels} channels")
        logger.info(f"Estimated processing time: {duration_sec / 60:.1f} minutes (varies by CPU/GPU speed)")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return False
    
    # Create preprocessor instance
    config = {}
    preprocessor = AudioPreprocessor(config)
    
    # Start GPU monitoring in a separate thread
    global monitoring_active
    monitoring_active = True
    monitor_thread = None
    
    if torch.cuda.is_available() and PYNVML_AVAILABLE:
        logger.info("Starting GPU monitoring...")
        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    # Start timer
    start_time = time.time()
    
    # Run Demucs
    logger.info(f"Running Demucs on {input_file}")
    result = preprocessor._run_demucs(input_file, output_file)
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Stop GPU monitoring
    monitoring_active = False
    if monitor_thread:
        monitor_thread.join(timeout=2)
    
    # Calculate GPU utilization statistics
    if gpu_utilization:
        avg_util = sum(gpu_utilization) / len(gpu_utilization)
        max_util = max(gpu_utilization)
        avg_mem = sum(gpu_memory_used) / len(gpu_memory_used) / 1024  # Convert to GB
        max_mem = max(gpu_memory_used) / 1024  # Convert to GB
        logger.info(f"\nGPU Utilization: Avg {avg_util:.1f}%, Max {max_util:.1f}%")
        logger.info(f"GPU Memory Usage: Avg {avg_mem:.1f} GB, Max {max_mem:.1f} GB")
    
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
    parser.add_argument("--duration", "-d", type=int, default=120, 
                      help="Maximum duration in seconds to process (default: 120 seconds, use 0 for full file)")
    args = parser.parse_args()
    
    # Convert duration of 0 to None (process full file)
    max_duration = args.duration if args.duration > 0 else None
    
    test_demucs_gpu(args.input, args.output_dir, max_duration)
