#!/usr/bin/env python3
# Speaker diarization module optimized for Swedish dialect separation
# Implements multi-stage approach with separate VAD and diarization
# 2025-04-23 - JS

import os
import sys
import time
import logging
import warnings
import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import datetime
import matplotlib.pyplot as plt
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Filter out specific warnings from torchaudio and other libraries
warnings.filterwarnings("ignore", message="torchaudio._backend.*has been deprecated")
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
warnings.filterwarnings("ignore", message="'audioop' is deprecated and slated for removal")

# Use the new import path for AudioMetaData
from torchaudio import AudioMetaData  # 2025-04-23 - JS

# Helper function to get colormap using the new API
def get_colormap(name):
    """Get a colormap using the new matplotlib API to avoid deprecation warnings.
    
    Args:
        name: Name of the colormap
        
    Returns:
        The requested colormap
    """
    return plt.colormaps[name]  # 2025-04-23 - JS


class SpeakerDiarizer:
    """
    Speaker diarization class that implements multi-stage approach
    optimized for Swedish dialect separation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the speaker diarizer with configuration.
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config or {}
        
        # Set up configuration parameters with defaults
        self.min_speakers = self.config.get('min_speakers', 2)
        self.max_speakers = self.config.get('max_speakers', 4)
        self.clustering_threshold = self.config.get('clustering_threshold', 0.65)
        self.use_gpu = self.config.get('use_gpu', torch.cuda.is_available())
        self.huggingface_token = self.config.get('huggingface_token', os.environ.get('HF_TOKEN'))
        self.batch_size = self.config.get('batch_size', 32)
        self.debug = self.config.get('debug', False)
        self.debug_dir = self.config.get('debug_dir', None)
        
        # Initialize pipelines
        self.diarization_pipeline = None
        self.vad_pipeline = None
        self.segmentation_pipeline = None
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Technical setup information should be at DEBUG level
        # 2025-04-24 -JS
        self.log(logging.DEBUG, "Speaker diarizer initialized")
        self.log(logging.DEBUG, f"Configuration: min_speakers={self.min_speakers}, max_speakers={self.max_speakers}, "
                             f"clustering_threshold={self.clustering_threshold}")
    
    def log(self, level, *messages, **kwargs):
        """
        Unified logging function.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            messages: Messages to log
            kwargs: Additional logging parameters
        """
        if level == logging.DEBUG:
            self.logger.debug(*messages, **kwargs)
        elif level == logging.INFO:
            self.logger.info(*messages, **kwargs)
        elif level == logging.WARNING:
            self.logger.warning(*messages, **kwargs)
        elif level == logging.ERROR:
            self.logger.error(*messages, **kwargs)
        elif level == logging.CRITICAL:
            self.logger.critical(*messages, **kwargs)
    
    def load_models(self):
        """
        Load diarization, VAD, and segmentation models.
        
        Returns:
            bool: True if models were loaded successfully, False otherwise
        """
        try:
            # Technical model loading details should be at DEBUG level
            # 2025-04-24 -JS
            self.log(logging.DEBUG, "Loading diarization model...")
            
            # Get diarization models from config or use defaults
            diarization_models = []
            
            # Try to get models from new config structure first
            if 'models' in self.config and 'diarization' in self.config['models']:
                # Add primary models
                if 'primary' in self.config['models']['diarization']:
                    diarization_models.extend(self.config['models']['diarization']['primary'])
                
                # Add fallback models
                if 'fallback' in self.config['models']['diarization']:
                    diarization_models.extend(self.config['models']['diarization']['fallback'])
            
            # Fallback to old config structure if needed
            elif 'models' in self.config:
                if 'primary' in self.config['models']:
                    diarization_models.extend(self.config['models']['primary'])
                if 'fallback' in self.config['models']:
                    diarization_models.extend(self.config['models']['fallback'])
            
            # Use hardcoded defaults if no models found in config
            if not diarization_models:
                diarization_models = [
                    "tensorlake/speaker-diarization-3.1",  # Preferred model for Swedish dialects
                    "pyannote/speaker-diarization-3.1"    # Fallback model
                ]
                
            self.log(logging.DEBUG, f"Using diarization models: {diarization_models}")  # 2025-04-24 -JS
            
            # Try loading models in order of preference
            for model_name in diarization_models:
                try:
                    self.log(logging.DEBUG, f"Trying to load diarization model: {model_name}")  # 2025-04-24 -JS
                    # Load the pipeline with the clustering threshold parameter
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=self.huggingface_token
                    )
                    
                    # Set the clustering threshold parameter for the pipeline
                    # In newer PyAnnote versions, we need to set this as a parameter of the pipeline
                    # rather than passing it to the apply() method
                    if hasattr(self.diarization_pipeline, "instantiate_params"):
                        self.log(logging.DEBUG, f"Setting clustering_threshold={self.clustering_threshold} for diarization pipeline")  # 2025-04-24 -JS
                        self.diarization_pipeline.instantiate_params = {
                            "clustering": {"threshold": self.clustering_threshold}
                        }
                    self.log(logging.DEBUG, f"Successfully loaded diarization model: {model_name}")  # 2025-04-24 -JS
                    break
                except Exception as e:
                    self.log(logging.WARNING, f"Failed to load {model_name}: {str(e)}")
                    continue
            
            # Check if any model was loaded
            if self.diarization_pipeline is None:
                self.log(logging.ERROR, "Failed to load any diarization model")
                return False
            
            # Optimize GPU usage if available
            if self.use_gpu and torch.cuda.is_available():
                self.log(logging.DEBUG, "Moving diarization pipeline to GPU...")  # 2025-04-24 -JS
                self.diarization_pipeline.to(torch.device("cuda"))
                
                # Enable TF32 for faster processing
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Set benchmark mode for faster processing with fixed input sizes
                torch.backends.cudnn.benchmark = True
                
                # Set batch size for better GPU utilization
                if hasattr(self.diarization_pipeline, "batch_size"):
                    self.diarization_pipeline.batch_size = self.batch_size
                    self.log(logging.DEBUG, f"Set batch_size to {self.batch_size}")  # 2025-04-24 -JS
                
                # Try to allocate more GPU memory
                torch.cuda.empty_cache()
                torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
            
            # Load VAD pipeline
            self.log(logging.DEBUG, "Loading Voice Activity Detection (VAD) model...")  # 2025-04-24 -JS
            
            # Get VAD models from config or use defaults
            vad_models = []
            
            # Try to get models from new config structure first
            if 'models' in self.config and 'vad' in self.config['models']:
                # Add primary models
                if 'primary' in self.config['models']['vad']:
                    vad_models.extend(self.config['models']['vad']['primary'])
                
                # Add fallback models
                if 'fallback' in self.config['models']['vad']:
                    vad_models.extend(self.config['models']['vad']['fallback'])
            
            # Fallback to old config structure if needed
            elif 'models' in self.config and 'additional' in self.config['models']:
                # Filter for VAD models in the additional list
                for model in self.config['models']['additional']:
                    if 'voice-activity-detection' in model or 'vad' in model.lower():
                        vad_models.append(model)
            
            # Use hardcoded defaults if no models found in config
            if not vad_models:
                vad_models = [
                    "pyannote/voice-activity-detection",
                    "pyannote/segmentation-3.0"
                ]
                
            self.log(logging.DEBUG, f"Using VAD models: {vad_models}")  # 2025-04-24 -JS
            
            # Try loading VAD models in order of preference
            self.vad_pipeline = None
            for model_name in vad_models:
                try:
                    self.log(logging.DEBUG, f"Trying to load VAD model: {model_name}")  # 2025-04-24 -JS
                    self.vad_pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=self.huggingface_token
                    )
                    self.log(logging.DEBUG, f"Successfully loaded VAD model: {model_name}")  # 2025-04-24 -JS
                    break
                except Exception as e:
                    self.log(logging.WARNING, f"Failed to load {model_name}: {str(e)}")
                    continue
            
            # Check if any VAD model was loaded
            if self.vad_pipeline is None:
                self.log(logging.WARNING, "Failed to load any VAD model, continuing without VAD")
            else:
                # Move VAD to GPU if available
                if self.use_gpu and torch.cuda.is_available():
                    self.vad_pipeline.to(torch.device("cuda"))
            
            # Try to load segmentation model
            try:
                self.log(logging.DEBUG, "Loading segmentation model...")  # 2025-04-24 -JS
                
                # Get segmentation models from config or use defaults
                segmentation_models = []
                
                # Try to get models from new config structure first
                if 'models' in self.config and 'segmentation' in self.config['models']:
                    # Add primary models
                    if 'primary' in self.config['models']['segmentation']:
                        segmentation_models.extend(self.config['models']['segmentation']['primary'])
                    
                    # Add fallback models
                    if 'fallback' in self.config['models']['segmentation']:
                        segmentation_models.extend(self.config['models']['segmentation']['fallback'])
                
                # Fallback to old config structure if needed
                elif 'models' in self.config and 'additional' in self.config['models']:
                    # Filter for segmentation models in the additional list
                    for model in self.config['models']['additional']:
                        if 'segmentation' in model:
                            segmentation_models.append(model)
                
                # Use hardcoded defaults if no models found in config
                if not segmentation_models:
                    segmentation_models = [
                        "pyannote/segmentation-3.0",
                        "HiTZ/pyannote-segmentation-3.0-RTVE"
                    ]
                    
                self.log(logging.DEBUG, f"Using segmentation models: {segmentation_models}")  # 2025-04-24 -JS
                
                # Try loading segmentation models in order of preference
                self.segmentation_pipeline = None
                for model_name in segmentation_models:
                    try:
                        self.log(logging.DEBUG, f"Trying to load segmentation model: {model_name}")  # 2025-04-24 -JS
                        self.segmentation_pipeline = Pipeline.from_pretrained(
                            model_name,
                            use_auth_token=self.huggingface_token
                        )
                        self.log(logging.DEBUG, f"Successfully loaded segmentation model: {model_name}")  # 2025-04-24 -JS
                        break
                    except Exception as e:
                        self.log(logging.WARNING, f"Failed to load {model_name}: {str(e)}")
                        continue
                        
                # If we couldn't load any model, raise an exception to be caught below
                if self.segmentation_pipeline is None:
                    raise ValueError("Failed to load any segmentation model")
                
                # Move segmentation to GPU if available
                if self.use_gpu and torch.cuda.is_available():
                    self.segmentation_pipeline.to(torch.device("cuda"))
                
                self.log(logging.DEBUG, "Segmentation model loaded successfully")  # 2025-04-24 -JS
            except Exception as e:
                self.log(logging.WARNING, f"Error loading segmentation model: {str(e)}")
                self.log(logging.WARNING, "Continuing without segmentation model")
                self.segmentation_pipeline = None
            
            self.log(logging.DEBUG, "Models loaded successfully")  # 2025-04-24 -JS
            return True
            
        except Exception as e:
            self.log(logging.ERROR, f"Error loading models: {str(e)}")
            return False
    
    def diarize(self, input_file, output_dir):
        """
        Perform speaker diarization on the input audio file.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save output files
            
        Returns:
            bool: True if diarization was successful, False otherwise
        """
        try:
            self.log(logging.INFO, f"Starting diarization of {input_file}")
            start_time = time.time()
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create debug directory if debug mode is enabled
            if self.debug and self.debug_dir:
                os.makedirs(self.debug_dir, exist_ok=True)
            
            # Load models if not already loaded
            if not self.diarization_pipeline or not self.vad_pipeline:
                if not self.load_models():
                    self.log(logging.ERROR, "Failed to load models")
                    return False
            
            # Generate base output filename
            base_output = os.path.splitext(os.path.basename(input_file))[0]
            
            # First run Voice Activity Detection if available
            speech_regions = None
            if self.vad_pipeline:
                self.log(logging.INFO, "Running Voice Activity Detection...")
                vad_start_time = time.time()
                
                # Run VAD to get speech regions
                vad_result = self.vad_pipeline(input_file)
                
                # Extract speech regions
                speech_regions = []
                for speech, _, _ in vad_result.itertracks(yield_label=True):
                    speech_regions.append({
                        "start": speech.start,
                        "end": speech.end
                    })
                
                vad_end_time = time.time()
                self.log(logging.DEBUG, f"Detected {len(speech_regions)} speech regions in {vad_end_time - vad_start_time:.2f} seconds")  # 2025-04-24 -JS
                
                # Save VAD results to file if debug mode is enabled
                if self.debug and self.debug_dir:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    vad_output_file = os.path.join(self.debug_dir, f"{base_output}.vad-{timestamp}.segments")
                    with open(vad_output_file, "w") as f:
                        for seg in speech_regions:
                            line = f"Speech from {seg['start']:.2f}s to {seg['end']:.2f}s"
                            f.write(line + "\n")
                    self.log(logging.DEBUG, f"Voice activity detection results saved to {vad_output_file}")  # 2025-04-24 -JS
            
            # Try different speaker counts
            all_results = {}
            successful_runs = []
            best_speaker_count = None
            max_segments = 0
            
            # Generate speaker counts to try
            speaker_counts = list(range(self.min_speakers, self.max_speakers + 1))
            
            for num_speakers in speaker_counts:
                try:
                    self.log(logging.DEBUG, f"Trying with num_speakers={num_speakers}")  # 2025-04-24 -JS
                    start_time_run = time.time()
                    
                    # Run diarization with the current speaker count
                    with ProgressHook() as hook:
                        # Configure the pipeline parameters
                        # Note: clustering_threshold is now set during pipeline initialization, not in apply()
                        # For newer PyAnnote versions, we need to set parameters differently
                        diarization = self.diarization_pipeline(
                            input_file, 
                            num_speakers=num_speakers, 
                            hook=hook
                        )
                    
                    # Process results for this run
                    segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segments.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    
                    # Generate output filename for this speaker count
                    run_output_file = os.path.join(output_dir, f"{base_output}.{num_speakers}speakers.segments")
                    
                    # Save segments to file
                    with open(run_output_file, "w") as f:
                        for seg in segments:
                            line = f"Speaker {seg['speaker']} from {seg['start']:.2f}s to {seg['end']:.2f}s"
                            f.write(line + "\n")
                    
                    # Calculate duration and stats
                    end_time_run = time.time()
                    duration_run = end_time_run - start_time_run
                    
                    # Store results
                    all_results[f"{num_speakers}speakers"] = {
                        "segments": segments,
                        "output_file": run_output_file,
                        "duration": duration_run,
                        "segment_count": len(segments),
                        "num_speakers": num_speakers
                    }
                    
                    # Check if this is the best run so far
                    if len(segments) > max_segments:
                        max_segments = len(segments)
                        best_speaker_count = num_speakers
                    
                    self.log(logging.INFO, f"Successfully completed diarization with {num_speakers} speakers")
                    self.log(logging.DEBUG, f"Found {len(segments)} speaker segments in {duration_run:.2f} seconds")  # 2025-04-24 -JS
                    self.log(logging.DEBUG, f"Results saved to {run_output_file}")  # 2025-04-24 -JS
                    
                    # Add to successful runs
                    successful_runs.append(num_speakers)
                    
                except Exception as e:
                    self.log(logging.ERROR, f"Error during diarization with {num_speakers} speakers: {str(e)}")
            
            # If no successful runs, try with auto speaker detection
            if not successful_runs:
                try:
                    self.log(logging.DEBUG, "Trying with auto speaker detection")  # 2025-04-24 -JS
                    start_time_run = time.time()
                    
                    # Run diarization with auto speaker detection
                    with ProgressHook() as hook:
                        # For newer PyAnnote versions, clustering_threshold is set during initialization
                        diarization = self.diarization_pipeline(
                            input_file, 
                            hook=hook
                        )
                    
                    # Process results
                    segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segments.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    
                    # Generate output filename
                    run_output_file = os.path.join(output_dir, f"{base_output}.auto.segments")
                    
                    # Save segments to file
                    with open(run_output_file, "w") as f:
                        for seg in segments:
                            line = f"Speaker {seg['speaker']} from {seg['start']:.2f}s to {seg['end']:.2f}s"
                            f.write(line + "\n")
                    
                    # Calculate duration and stats
                    end_time_run = time.time()
                    duration_run = end_time_run - start_time_run
                    
                    self.log(logging.INFO, "Successfully completed diarization with auto speaker detection")
                    self.log(logging.DEBUG, f"Found {len(segments)} speaker segments in {duration_run:.2f} seconds")  # 2025-04-24 -JS
                    self.log(logging.DEBUG, f"Results saved to {run_output_file}")  # 2025-04-24 -JS
                    
                except Exception as e:
                    self.log(logging.ERROR, f"Error during auto diarization: {str(e)}")
            
            # Print timing information
            end_time = time.time()
            total_duration = end_time - start_time
            self.log(logging.DEBUG, f"Total diarization time: {total_duration:.2f} seconds")  # 2025-04-24 -JS
            
            # Create a summary file
            summary_file = os.path.join(output_dir, f"{base_output}.diarization_summary.txt")
            with open(summary_file, "w") as f:
                f.write(f"Diarization Summary for {input_file}\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total processing time: {total_duration:.2f} seconds\n\n")
                
                if best_speaker_count:
                    f.write(f"Best speaker count: {best_speaker_count} (with {max_segments} segments)\n\n")
                
                f.write("Results by speaker count:\n")
                for count in speaker_counts:
                    if f"{count}speakers" in all_results:
                        result = all_results[f"{count}speakers"]
                        f.write(f"  {count} speakers: {result['segment_count']} segments in {result['duration']:.2f} seconds\n")
                    else:
                        f.write(f"  {count} speakers: Failed\n")
            
            self.log(logging.DEBUG, f"Diarization summary saved to {summary_file}")  # 2025-04-24 -JS
            
            # Initialize diarization segments if not already done
            if not hasattr(self, 'diarization_segments'):
                self.diarization_segments = []
                
                # Try to extract segments from the best result
                if 'best_result' in locals() and best_result and 'diarization' in best_result:
                    diarization = best_result['diarization']
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segment = {
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": f"SPEAKER_{speaker.split('_')[-1].zfill(2)}",
                            "text": ""
                        }
                        self.diarization_segments.append(segment)
                elif hasattr(self, 'diarization_result') and self.diarization_result:
                    # Extract from the main diarization result if available
                    for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
                        segment = {
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": f"SPEAKER_{speaker.split('_')[-1].zfill(2)}",
                            "text": ""
                        }
                        self.diarization_segments.append(segment)
            
            return True
            
        except Exception as e:
            self.log(logging.ERROR, f"Error during diarization: {str(e)}")
            return False
    
    def get_diarization_result(self):
        """
        Get the diarization result in a format suitable for SRT generation.
        
        Returns:
            list: List of diarization segments with start, end, speaker, and text fields
        """
        if hasattr(self, 'diarization_segments') and self.diarization_segments:
            return self.diarization_segments
        else:
            self.log(logging.WARNING, "No diarization segments available")
            return []
    
    def save_diarization_result(self, output_file):
        """
        Save the diarization result to a JSON file.
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            if not hasattr(self, 'diarization_segments') or not self.diarization_segments:
                self.log(logging.WARNING, "No diarization segments to save")
                return False
                
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save segments to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.diarization_segments, f, indent=2)
                
            self.log(logging.DEBUG, f"Saved {len(self.diarization_segments)} diarization segments to {output_file}")  # 2025-04-24 -JS
            return True
            
        except Exception as e:
            self.log(logging.ERROR, f"Error saving diarization result: {str(e)}")
            return False
