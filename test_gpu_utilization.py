#!/usr/bin/env python
# Test script to measure GPU utilization during pyannote processing
# 2025-04-24 -JS

import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread

# Force TF32 acceleration for better GPU performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the SpeakerDiarizer
from src.audio_processing.diarization import SpeakerDiarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_test')

class GPUMonitor:
    """Monitor GPU utilization during processing."""
    
    def __init__(self, interval=0.5):
        """Initialize the GPU monitor.
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.running = False
        self.gpu_utilization = []
        self.gpu_memory = []
        self.timestamps = []
        self.start_time = None
        
    def start(self):
        """Start monitoring GPU utilization."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot monitor GPU")
            return
            
        self.running = True
        self.start_time = time.time()
        self.gpu_utilization = []
        self.gpu_memory = []
        self.timestamps = []
        
        # Start monitoring thread
        self.thread = Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring GPU utilization."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
            
    def _monitor(self):
        """Monitor GPU utilization in a separate thread."""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                logger.warning("No NVIDIA devices found")
                return
                
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while self.running:
                try:
                    # Get GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    
                    # Get memory utilization
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_mb = memory.used / (1024 * 1024)
                    mem_total_mb = memory.total / (1024 * 1024)
                    mem_percent = (memory.used / memory.total) * 100
                    
                    # Record measurements
                    self.gpu_utilization.append(gpu_util)
                    self.gpu_memory.append(mem_percent)
                    self.timestamps.append(time.time() - self.start_time)
                    
                    logger.debug(f"GPU: {gpu_util}%, Memory: {mem_used_mb:.0f}MB / {mem_total_mb:.0f}MB ({mem_percent:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error monitoring GPU: {str(e)}")
                    
                time.sleep(self.interval)
                
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.error("pynvml not installed, cannot monitor GPU. Install with: pip install nvidia-ml-py")
        except Exception as e:
            logger.error(f"Error initializing GPU monitoring: {str(e)}")
            
    def plot(self, output_file=None):
        """Plot GPU utilization over time.
        
        Args:
            output_file: Path to save the plot, if None, display the plot
        """
        if not self.timestamps:
            logger.warning("No GPU data collected")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot GPU utilization
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.gpu_utilization, 'b-')
        plt.title('GPU Utilization')
        plt.ylabel('Utilization (%)')
        plt.grid(True)
        
        # Plot memory utilization
        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.gpu_memory, 'r-')
        plt.title('GPU Memory Usage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (%)')
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved GPU utilization plot to {output_file}")
        else:
            plt.show()

def test_gpu_utilization(audio_file, config_file=None, output_dir=None, max_duration=60):
    """Test GPU utilization during pyannote processing.
    
    Args:
        audio_file: Path to audio file to process
        config_file: Path to config file (optional)
        output_dir: Directory to save output files (optional)
        max_duration: Maximum duration in seconds to process (default: 60)
    """
    # Load configuration
    config = None
    if config_file and os.path.exists(config_file):
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
    
    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize GPU monitor
    monitor = GPUMonitor(interval=0.2)  # Sample every 200ms
    
    try:
        # Initialize diarizer
        logger.info("Initializing SpeakerDiarizer...")
        diarizer = SpeakerDiarizer(config)
        
        # Load models
        logger.info("Loading models...")
        diarizer.load_models()
        
        # Start GPU monitoring
        logger.info("Starting GPU monitoring...")
        monitor.start()
        
        # Process audio file
        logger.info(f"Processing audio file: {audio_file}")
        
        # Create a temporary file with only a portion of the audio if max_duration is specified
        temp_file = None
        if max_duration > 0:
            from pydub import AudioSegment
            try:
                logger.info(f"Loading first {max_duration} seconds of audio...")
                audio = AudioSegment.from_file(audio_file)
                # Trim to max_duration seconds
                if len(audio) > max_duration * 1000:  # pydub works in milliseconds
                    audio = audio[:max_duration * 1000]
                    
                # Create a temporary file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                logger.info(f"Saving trimmed audio to {temp_file.name}")
                audio.export(temp_file.name, format="wav")
                
                # Use the temporary file for processing
                process_file = temp_file.name
                logger.info(f"Processing first {max_duration} seconds of audio")
            except Exception as e:
                logger.error(f"Error trimming audio: {str(e)}")
                process_file = audio_file
        else:
            process_file = audio_file
            
        # Process the audio file
        start_time = time.time()
        diarization = diarizer.process_audio(process_file)
        processing_time = time.time() - start_time
        
        # Clean up temporary file if created
        if temp_file:
            try:
                os.unlink(temp_file.name)
                logger.info(f"Removed temporary file {temp_file.name}")
            except Exception as e:
                logger.warning(f"Error removing temporary file: {str(e)}")
        
        # Stop GPU monitoring
        monitor.stop()
        
        # Log results
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        if diarization:
            # Get number of speakers
            speakers = set()
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
            
            logger.info(f"Detected {len(speakers)} speakers")
            
            # Print first few segments
            logger.info("First 5 segments:")
            for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                if i >= 5:
                    break
                logger.info(f"  {segment.start:.2f}s - {segment.end:.2f}s: {speaker}")
        
        # Plot GPU utilization
        if output_dir:
            plot_file = os.path.join(output_dir, f"gpu_utilization_{os.path.basename(audio_file)}.png")
            monitor.plot(output_file=plot_file)
        else:
            monitor.plot()
            
        # Calculate statistics
        if monitor.gpu_utilization:
            avg_util = np.mean(monitor.gpu_utilization)
            max_util = np.max(monitor.gpu_utilization)
            avg_mem = np.mean(monitor.gpu_memory)
            max_mem = np.max(monitor.gpu_memory)
            
            logger.info(f"GPU Utilization: Avg={avg_util:.1f}%, Max={max_util:.1f}%")
            logger.info(f"GPU Memory Usage: Avg={avg_mem:.1f}%, Max={max_mem:.1f}%")
            
            # Save statistics to file
            if output_dir:
                stats_file = os.path.join(output_dir, f"gpu_stats_{os.path.basename(audio_file)}.txt")
                with open(stats_file, 'w') as f:
                    f.write(f"Processing time: {processing_time:.2f} seconds\n")
                    f.write(f"GPU Utilization: Avg={avg_util:.1f}%, Max={max_util:.1f}%\n")
                    f.write(f"GPU Memory Usage: Avg={avg_mem:.1f}%, Max={max_mem:.1f}%\n")
                logger.info(f"Saved statistics to {stats_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Ensure GPU monitoring is stopped
        monitor.stop()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test GPU utilization during pyannote processing")
    parser.add_argument("audio_file", help="Path to audio file to process")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    parser.add_argument("--max-duration", "-m", type=int, default=30, 
                      help="Maximum duration in seconds to process (default: 30)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run the test
    test_gpu_utilization(args.audio_file, args.config, args.output_dir, args.max_duration)

if __name__ == "__main__":
    main()
