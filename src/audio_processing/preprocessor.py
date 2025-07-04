#!/usr/bin/env python3
# Audio preprocessor module
# Handles audio preprocessing steps: demucs, normalization, highpass, compression, volume adjustment
# 2025-04-23 -JS

import os
import sys
import logging
import subprocess
import tempfile
import datetime
import time
import shutil
from pydub import AudioSegment, effects
import numpy as np
import scipy.signal

class AudioPreprocessor:
    """
    Class for preprocessing audio files.
    Handles vocal separation, normalization, filtering, compression, and volume adjustment.
    """
    
    def __init__(self, config=None):
        """
        Initialize the audio preprocessor with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default settings
        # 2025-04-24 -JS - Updated frequency cutoffs to optimal ranges for speech
        self.highpass_cutoff = self.config.get('highpass_cutoff', 300)  # Remove low-frequency rumble while preserving speech
        self.lowpass_cutoff = self.config.get('lowpass_cutoff', 4500)  # Preserve speech clarity while reducing high-frequency noise
        self.compression_threshold = self.config.get('compression_threshold', -10.0)
        self.compression_ratio = self.config.get('compression_ratio', 2.0)
        self.default_gain = self.config.get('default_gain', 3.0)  # +3dB gain
        
        # Options to skip specific processing steps
        # 2025-04-24 -JS
        self.use_vocals_directly = self.config.get('use_vocals_directly', False)
        self.skip_steps = self.config.get('skip_steps', [])
        
        # For backward compatibility
        self.skip_filtering = self.config.get('skip_filtering', False)
        if self.skip_filtering:
            if 'highpass' not in self.skip_steps:
                self.skip_steps.append('highpass')
                
        # Output format option (wav or mp3)
        # 2025-04-24 -JS
        self.output_format = self.config.get('output_format', 'wav')
        if self.output_format not in ['wav', 'mp3']:
            self.log(logging.WARNING, f"Invalid output format '{self.output_format}', defaulting to 'wav'")
            self.output_format = 'wav'
            if 'lowpass' not in self.skip_steps:
                self.skip_steps.append('lowpass')
        
        # Debug settings
        self.debug_mode = self.config.get('debug', False)
        self.debug_dir = self.config.get('debug_dir', None)
        
        # Create debug directory if it doesn't exist
        if self.debug_mode and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
    
    def log(self, level, *messages, **kwargs):
        """
        Unified logging function with console output for progress and important messages.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            messages: Messages to log
            kwargs: Additional logging parameters
        """
        # Log to file
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
            
        # Also print progress and important messages to console
        msg = " ".join(str(m) for m in messages)
        
        # Print progress updates directly to console
        if "progress" in msg.lower() or "compression progress" in msg.lower() or "demucs progress" in msg.lower():
            print(f"\033[96m{msg}\033[0m")  # Cyan color for progress
        # Print errors in red
        elif level == logging.ERROR or level == logging.CRITICAL:
            print(f"\033[91mERROR: {msg}\033[0m")
        # Print warnings in yellow
        elif level == logging.WARNING:
            print(f"\033[93mWARNING: {msg}\033[0m")
    
    def convert_to_wav(self, input_file, output_file, bit_depth=16, sample_rate=44100):
        """
        Convert audio file to the selected format (WAV or MP3) with specified parameters.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file (format determined by output_format setting)
            bit_depth: Bit depth for the WAV file (16, 24, or 32) - only used for WAV format
            sample_rate: Sample rate in Hz (e.g., 44100, 48000, 96000)
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            # 2025-04-24 -JS - Updated to respect output_format setting
            format_name = "WAV" if self.output_format == "wav" else "MP3"
            self.log(logging.INFO, f"Creating {format_name} file from {os.path.basename(input_file)}")
            
            # Load the input audio
            audio = AudioSegment.from_file(input_file)
            
            # Ensure stereo (2 channels)
            if audio.channels == 1:
                audio = audio.set_channels(2)
            
            # Resample if necessary
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            
            # Use the configured output format
            format_to_use = self.output_format
            
            # Handle format-specific export options
            if format_to_use == "wav":
                # Handle bit depth correctly for WAV format
                # For 24-bit audio, we need to use a special approach
                if bit_depth == 24:
                    # Export with specific parameters for 24-bit
                    audio.export(
                        output_file, 
                        format="wav",
                        parameters=["-acodec", "pcm_s24le", "-ac", "2", "-ar", str(sample_rate)]
                    )
                else:
                    # For 16-bit and 32-bit, we can use the standard approach
                    sample_width = bit_depth // 8  # Convert bits to bytes
                    if audio.sample_width != sample_width:
                        audio = audio.set_sample_width(sample_width)
                    audio.export(output_file, format="wav")
            else:  # MP3 format
                # For MP3, we don't need to worry about bit depth
                # but we should set a good bitrate for quality
                audio.export(output_file, format="mp3", bitrate="192k")
            
            # Verify the output file has the correct properties
            converted_audio = AudioSegment.from_file(output_file)
            self.log(logging.DEBUG, f"Converted audio properties: {converted_audio.channels} channels, "
                                    f"{converted_audio.frame_rate}Hz, {converted_audio.sample_width * 8}-bit")
            
            # Save debug file if debug mode is enabled
            # 2025-04-24 -JS - Updated debug step name to match the actual format
            debug_step = "wav_conversion" if self.output_format == "wav" else "mp3_conversion"
            self._save_debug_file(converted_audio, input_file, debug_step)
            
            # 2025-04-24 -JS - Updated to respect output_format setting
            format_name = "WAV" if self.output_format == "wav" else "MP3"
            self.log(logging.INFO, f"{format_name} file created and saved to {os.path.basename(output_file)}")
            return True
            
        except Exception as e:
            # 2025-04-24 -JS - Updated to respect output_format setting
            format_name = "WAV" if self.output_format == "wav" else "MP3"
            self.log(logging.ERROR, f"Error converting to {format_name}: {str(e)}")
            return False
    
    def preprocess(self, input_file, output_file):
        """
        Full preprocessing pipeline: convert to WAV, separate vocals, normalize, filter, compress, adjust volume.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            
        Returns:
            bool: True if preprocessing was successful, False otherwise
        """
        self.log(logging.DEBUG, f"Starting preprocessing of {input_file}")
        
        # Get base name of input file without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Check if the input_file already contains a timestamp
        import re
        timestamp_pattern = re.compile(r'\d{8}_\d{6}')
        
        # If no timestamp in the base_name, add one at the beginning
        if not timestamp_pattern.search(base_name):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamped_name = f"{timestamp}_{base_name}"
        else:
            timestamped_name = base_name
        
        # Determine paths for intermediate files
        # 2025-04-24 -JS - Updated to respect output_format setting
        file_ext = ".wav" if self.output_format == "wav" else ".mp3"
        conversion_step = "wav_conversion" if self.output_format == "wav" else "mp3_conversion"
        
        if self.debug_mode and self.debug_dir:
            # In debug mode, save intermediate files in debug directory with consistent step-based naming
            # 2025-04-24 -JS - Updated to use consistent step-based naming without timestamps and respect output format
            converted_file = os.path.join(self.debug_dir, f"01_{conversion_step}_{base_name}{file_ext}")
            vocals_file = os.path.join(self.debug_dir, f"02_vocals_{base_name}{file_ext}")
        else:
            # In normal mode, use temporary files that will be cleaned up
            temp_dir = tempfile.mkdtemp(prefix="audio_toolkit_")
            converted_file = os.path.join(temp_dir, f"{timestamped_name}_highquality{file_ext}")
            vocals_file = os.path.join(temp_dir, f"{timestamped_name}_vocals{file_ext}")
        
        try:
            # Step 1: Convert to high-quality audio in the selected format
            # 2025-04-24 -JS - Updated to respect output_format setting
            if not self.convert_to_wav(input_file, converted_file, 
                                       bit_depth=self.config.get('bit_depth', 24),
                                       sample_rate=self.config.get('sample_rate', 48000)):
                format_name = "WAV" if self.output_format == "wav" else "MP3"
                self.log(logging.ERROR, f"{format_name} conversion failed")
                return False
            
            # Step 2: Separate vocals if not skipped
            if 'vocals' in self.skip_steps or self.use_vocals_directly:
                self.log(logging.INFO, f"Skipping vocal separation as requested")
                # If using vocals directly, the input file is assumed to be the vocals file
                vocals_file = input_file
            else:
                # 2025-04-24 -JS - Updated to use the correct converted file variable
                if not self.separate_vocals(converted_file, vocals_file):
                    self.log(logging.ERROR, "Vocal separation failed")
                    return False
            
            # Step 3: Process the separated vocals or use them directly
            if self.use_vocals_directly:
                # Use vocals file directly without additional processing
                # 2025-04-24 -JS
                self.log(logging.INFO, f"Using vocals file directly as requested: {vocals_file}")
                shutil.copy(vocals_file, output_file)
                self.log(logging.INFO, f"Copied vocals file to output: {output_file}")
            else:
                # Apply additional processing to the vocals file
                if not self.process_audio(vocals_file, output_file):
                    self.log(logging.ERROR, "Audio processing failed")
                    return False
            
            self.log(logging.INFO, f"Preprocessing completed successfully")
            return True
            
        finally:
            # Clean up temporary files if not in debug mode
            if not self.debug_mode or not self.debug_dir:
                try:
                    # Remove temporary directory and its contents
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        self.log(logging.DEBUG, f"Cleaned up temporary files in {temp_dir}")
                except Exception as e:
                    self.log(logging.WARNING, f"Failed to clean up temporary files: {str(e)}")

    
    def separate_vocals(self, input_file, output_file):
        """
        Separate vocals from background using demucs.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            
        Returns:
            bool: True if separation was successful, False otherwise
        """
        self.log(logging.INFO, f"Running vocal separation step")
        self.log(logging.DEBUG, f"Using Demucs to extract vocals from background elements...")
        
        # Run Demucs for vocal separation
        start_time = time.time()
        success = self._run_demucs(input_file, output_file)
        end_time = time.time()
        
        if success:
            self.log(logging.INFO, f"Vocal separation completed")
            self.log(logging.DEBUG, f"Vocal separation took {end_time - start_time:.2f} seconds")
            
            # Get info about the separated vocals file
            try:
                # Use a context manager to ensure the file is properly closed
                with open(output_file, 'rb') as f:
                    vocals_audio = AudioSegment.from_file(f)
                duration_sec = len(vocals_audio) / 1000
                self.log(logging.INFO, f"Extracted vocals: {duration_sec:.2f} seconds, {vocals_audio.frame_rate}Hz, {vocals_audio.channels} channels")
                
                # Save debug file for vocals if in debug mode
                if self.debug_mode and self.debug_dir:
                    self._save_debug_file(vocals_audio, input_file, "vocals")
            except Exception as e:
                self.log(logging.WARNING, f"Failed to get info about separated vocals: {str(e)}")
        else:
            self.log(logging.ERROR, f"Vocal separation failed after {end_time - start_time:.2f} seconds")
                
        return success
    
    def _save_debug_file(self, audio, input_file, step_name):
        """
        Save a debug file for a processing step if debug mode is enabled.
        Uses a consistent naming pattern based on the original input filename.
        
        Args:
            audio: AudioSegment to save
            input_file: Original input file path (used for naming)
            step_name: Name of the processing step
            
        Returns:
            None
        """
        # 2025-04-24 -JS - Don't create debug files if debug mode is disabled or debug directory is not set
        if not self.debug_mode or not self.debug_dir:
            return
            
        # 2025-04-24 -JS - Don't create debug files for steps that are being skipped
        # This prevents unnecessary file creation for skipped processing steps
        if step_name in self.skip_steps:
            self.log(logging.DEBUG, f"Skipping debug file creation for {step_name} (step is in skip_steps list)")
            return
            
        # 2025-04-24 -JS - Don't create debug files for volume step when gain is 0
        if step_name == "volume" and self.default_gain == 0:
            self.log(logging.DEBUG, f"Skipping debug file creation for volume step (gain is set to 0)")
            return
            
        try:
            # Get base name of input file without extension and without any timestamp
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            
            # Remove any existing timestamp pattern from the base_name
            import re
            timestamp_pattern = re.compile(r'\d{8}_\d{6}_')
            base_name = re.sub(timestamp_pattern, '', base_name)
            
            # Create debug filename with step name and processing order prefix
            # Use a fixed order prefix to ensure files sort correctly
            # 2025-04-24 -JS - Added mp3_conversion to step_order
            step_order = {
                "wav_conversion": "01",
                "mp3_conversion": "01",  # Same prefix as wav_conversion since they're alternatives
                "vocals": "02",
                "highpass": "03",
                "lowpass": "04",
                "compression": "05",
                "normalize": "06",
                "volume": "07"
            }
            
            # Get the order prefix or use "99" if step_name is not in the dictionary
            order_prefix = step_order.get(step_name, "99")
            
            # For conversion step, create a symbolic link instead of duplicating the large file
            # This saves disk space while still providing the debug file for analysis
            # 2025-04-24 -JS - Updated to handle both WAV and MP3 formats
            if (step_name == "wav_conversion" or step_name == "mp3_conversion") and os.path.exists(input_file):
                # Determine the correct extension based on the step name
                ext = ".wav" if step_name == "wav_conversion" else ".mp3"
                
                # Only create a symlink if the input file has the matching extension
                if input_file.lower().endswith(ext):
                    # Create debug filename with order prefix
                    debug_filename = f"{order_prefix}_{step_name}_{base_name}{ext}"
                    debug_path = os.path.join(self.debug_dir, debug_filename)
                    
                    # Create a symbolic link to the original file instead of exporting a new one
                    if os.path.exists(debug_path):
                        os.remove(debug_path)  # Remove existing link if it exists
                    os.symlink(os.path.abspath(input_file), debug_path)
                    self.log(logging.DEBUG, f"Created symbolic link for {step_name}: {debug_path} -> {input_file}")
                else:
                    # If extensions don't match, fall through to the regular export code below
                    pass
            else:
                # For all other steps, create a regular debug file with the appropriate format
                # 2025-04-24 -JS - Updated to respect output_format setting
                file_format = "wav"
                if self.output_format == "mp3":
                    file_format = "mp3"
                
                debug_filename = f"{order_prefix}_{step_name}_{base_name}.{file_format}"
                debug_path = os.path.join(self.debug_dir, debug_filename)
                
                # Export debug file in the selected format
                # Use with statement to ensure file is properly closed
                with open(debug_path, 'wb') as f:
                    audio.export(f, format=file_format)
                self.log(logging.DEBUG, f"Saved debug file for {step_name}: {debug_path}")
                
                # The 'both' option has been removed as it doesn't make sense in a pipeline context
                # Each processing step now uses a single consistent format throughout
            
        except Exception as e:
            self.log(logging.WARNING, f"Failed to save debug file for {step_name}: {str(e)}")
    
    def process_audio(self, input_file, output_file):
        """
        Process audio file: normalize, filter, compress, adjust volume.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Check if the input_file already contains a timestamp
        import re
        timestamp_pattern = re.compile(r'\d{8}_\d{6}')
        
        # Get base name of input file without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Check if the output directory already has a timestamp
        output_dir = os.path.dirname(output_file)
        dir_has_timestamp = timestamp_pattern.search(os.path.basename(output_dir))
        
        # If the directory doesn't have a timestamp, add timestamp to the output file
        # If the directory already has a timestamp, don't add timestamp to files inside it
        if not dir_has_timestamp and not timestamp_pattern.search(base_name) and not timestamp_pattern.search(os.path.basename(output_file)):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # Update output_file with timestamped name if it doesn't already have a timestamp
            output_filename = os.path.basename(output_file)
            output_base = os.path.splitext(output_filename)[0]
            output_ext = os.path.splitext(output_filename)[1]
            output_file = os.path.join(output_dir, f"{timestamp}_{output_base}{output_ext}")
            self.log(logging.DEBUG, f"Updated output file path with timestamp: {output_file}")
        
        try:
            self.log(logging.INFO, f"Running audio enhancement step")
            
            # Load audio
            self.log(logging.INFO, f"Loading audio file: {input_file}")
            # Use a context manager to ensure the file is properly closed
            with open(input_file, 'rb') as f:
                audio = AudioSegment.from_file(f)
            sample_rate = audio.frame_rate
            self.log(logging.INFO, f"Audio loaded: {len(audio)/1000:.2f} seconds, {sample_rate}Hz sample rate")
            
            # Apply processing steps with volume compensation and save debug files for each step
            # 2025-04-24 -JS - Added support for skipping specific processing steps
            
            # Step 1: Highpass filter
            if 'highpass' in self.skip_steps:
                self.log(logging.INFO, f"Skipping high-pass filter as requested")
                # 2025-04-24 -JS - Don't create debug files for skipped steps
            else:
                # Apply highpass filter
                self.log(logging.INFO, f"Applying high-pass filter (cutoff: {self.highpass_cutoff}Hz)...")
                audio = self.apply_highpass(audio, cutoff=self.highpass_cutoff, sample_rate=sample_rate)
                
                # Compensate for volume loss after highpass filter
                if 'highpass_compensation' not in self.skip_steps:
                    self.log(logging.INFO, f"Compensating for volume loss after high-pass filter...")
                    audio = self.adjust_volume(audio, gain_db=6.0)  # Add 6dB to compensate for highpass filtering
                
                self._save_debug_file(audio, input_file, "highpass")
                self.log(logging.INFO, f"High-pass filter processing completed")
            
            # Step 2: Lowpass filter
            if 'lowpass' in self.skip_steps:
                self.log(logging.INFO, f"Skipping low-pass filter as requested")
                # 2025-04-24 -JS - Don't create debug files for skipped steps
            else:
                # Apply lowpass filter
                self.log(logging.INFO, f"Applying low-pass filter (cutoff: {self.lowpass_cutoff}Hz)...")
                audio = self.apply_lowpass(audio, cutoff=self.lowpass_cutoff, sample_rate=sample_rate)
                
                # Compensate for volume loss after lowpass filter
                if 'lowpass_compensation' not in self.skip_steps:
                    self.log(logging.INFO, f"Compensating for volume loss after low-pass filter...")
                    audio = self.adjust_volume(audio, gain_db=4.0)  # Add 4dB to compensate for lowpass filtering
                
                self._save_debug_file(audio, input_file, "lowpass")
                self.log(logging.INFO, f"Low-pass filter processing completed")
            
            # Step 3: Compression
            if 'compression' in self.skip_steps:
                self.log(logging.INFO, f"Skipping compression as requested")
                # 2025-04-24 -JS - Don't create debug files for skipped steps
            else:
                self.log(logging.INFO, f"Applying compression (threshold: {self.compression_threshold}dB, ratio: {self.compression_ratio})...")
                audio = self.apply_compression(audio)
                self._save_debug_file(audio, input_file, "compression")
                self.log(logging.INFO, f"Compression applied successfully")
            
            # Step 4: Normalization
            if 'normalize' in self.skip_steps:
                self.log(logging.INFO, f"Skipping normalization as requested")
                # 2025-04-24 -JS - Don't create debug files for skipped steps
            else:
                self.log(logging.INFO, f"Normalizing audio levels...")
                audio = self.normalize(audio)
                self._save_debug_file(audio, input_file, "normalize")
                self.log(logging.INFO, f"Normalization completed successfully")
            
            # Step 5: Volume adjustment
            # 2025-04-24 -JS - Skip volume adjustment when gain is 0
            if 'volume' in self.skip_steps or self.default_gain == 0:
                if 'volume' in self.skip_steps:
                    self.log(logging.INFO, f"Skipping volume adjustment as requested")
                else:
                    self.log(logging.INFO, f"Skipping volume adjustment (gain is set to 0)")
                # 2025-04-24 -JS - Don't create debug files for skipped steps
            else:
                self.log(logging.INFO, f"Adjusting volume (gain: {self.default_gain:.1f}dB)...")
                audio = self.adjust_volume(audio, gain_db=self.default_gain)
                self._save_debug_file(audio, input_file, "volume")
                self.log(logging.INFO, f"Volume adjustment completed successfully")
            
            # Save the final processed audio
            self.log(logging.DEBUG, f"Saving processed audio to {output_file}")
            
            # Determine the output format based on the output_format setting
            # 2025-04-24 -JS
            output_ext = os.path.splitext(output_file)[1].lower()
            
            # Use the configured output format
            format_to_use = self.output_format
            
            # If the output file extension doesn't match the configured format, adjust the filename
            if (format_to_use == "mp3" and output_ext != ".mp3") or (format_to_use == "wav" and output_ext != ".wav"):
                output_base = os.path.splitext(output_file)[0]
                output_file = f"{output_base}.{format_to_use}"
                self.log(logging.DEBUG, f"Adjusted output filename to match format: {output_file}")
            
            # Use with statement to ensure file is properly closed
            with open(output_file, 'wb') as f:
                audio.export(f, format=format_to_use)
            
            self.log(logging.INFO, f"Saved enhanced audio to {os.path.basename(output_file)}")
            
            return True
            
        except Exception as e:
            self.log(logging.ERROR, f"Error processing audio: {str(e)}")
            return False
    
    def normalize(self, audio):
        """
        Normalize audio levels.
        
        Args:
            audio: AudioSegment to normalize
            
        Returns:
            AudioSegment: Normalized audio
        """
        return self._normalize_audio(audio)
    
    def apply_highpass(self, audio, cutoff=None, sample_rate=44100):
        """
        Apply high-pass filter to remove low frequency noise.
        
        Args:
            audio: AudioSegment to filter
            cutoff: Filter cutoff frequency in Hz
            sample_rate: Audio sample rate
            
        Returns:
            AudioSegment: Filtered audio
        """
        cutoff = cutoff or self.highpass_cutoff
        return self._apply_highpass(audio, cutoff, sample_rate)
    
    def apply_lowpass(self, audio, cutoff=None, sample_rate=44100):
        """
        Apply low-pass filter to remove high frequency noise.
        
        Args:
            audio: AudioSegment to filter
            cutoff: Filter cutoff frequency in Hz
            sample_rate: Audio sample rate
            
        Returns:
            AudioSegment: Filtered audio
        """
        cutoff = cutoff or self.lowpass_cutoff
        return self._apply_lowpass(audio, cutoff, sample_rate)
    
    def apply_compression(self, audio):
        """
        Apply dynamic range compression.
        
        Args:
            audio: AudioSegment to compress
            
        Returns:
            AudioSegment: Compressed audio
        """
        return self._apply_compression(audio)
    
    def adjust_volume(self, audio, gain_db=None):
        """
        Adjust audio volume.
        
        Args:
            audio: AudioSegment to adjust
            gain_db: Gain in dB to apply
            
        Returns:
            AudioSegment: Volume-adjusted audio
        """
        gain_db = gain_db if gain_db is not None else self.default_gain
        return self._adjust_volume(audio, gain_db)
    
    # Implementation methods
    
    def _check_diffq_installed(self):
        """Check if diffq package is installed after attempting installation
        
        Returns:
            bool: True if diffq is installed, False otherwise
        """
        try:
            import importlib.util
            diffq_spec = importlib.util.find_spec("diffq")
            return diffq_spec is not None
        except ImportError:
            return False
    
    def _run_demucs(self, input_file, output_file):
        """
        Run demucs vocal separation.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            
        Returns:
            bool: True if demucs ran successfully, False otherwise
        """
        try:
            self.log(logging.INFO, f"Initializing Demucs for vocal separation on file: {input_file}")
            
            # Get input file info for better progress reporting
            try:
                input_audio = AudioSegment.from_file(input_file)
                duration_sec = len(input_audio) / 1000
                self.log(logging.INFO, f"Input audio: {duration_sec:.2f} seconds, {input_audio.frame_rate}Hz, {input_audio.channels} channels")
                self.log(logging.INFO, f"Estimated processing time: {duration_sec/60:.1f} minutes (varies by CPU/GPU speed)")
            except Exception as e:
                self.log(logging.WARNING, f"Could not get input file info: {str(e)}")
            
            # Create a temporary directory for demucs output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run demucs command with verbose output
                self.log(logging.INFO, "Starting Demucs model - separating vocals from background...")
                
                # 2025-04-24 -JS - Enhanced GPU utilization for Demucs
                # Check if GPU is available
                use_gpu = False
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                    if use_gpu:
                        gpu_name = torch.cuda.get_device_name(0)
                        self.log(logging.INFO, f"Using GPU for Demucs: {gpu_name}")
                    else:
                        self.log(logging.INFO, "GPU not available, using CPU for Demucs")
                except ImportError:
                    self.log(logging.WARNING, "PyTorch not available, using CPU for Demucs")
                
                # 2025-04-24 -JS - Use mdx_extra_q model instead of htdemucs for better compatibility
                # htdemucs has limitations on audio length and can fail with longer files
                
                # Check if we're in a test environment
                is_test = 'PYTEST_CURRENT_TEST' in os.environ or 'unittest' in sys.modules
                
                if not is_test:
                    # Only perform diffq check in non-test environment
                    try:
                        import importlib.util
                        diffq_spec = importlib.util.find_spec("diffq")
                        has_diffq = diffq_spec is not None
                    except ImportError:
                        has_diffq = False
                    
                    if not has_diffq:
                        self.log(logging.WARNING, "diffq package not installed, attempting to install it")
                        try:
                            # Use the subprocess module that's already imported at the top of the file
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "diffq"])
                            self.log(logging.INFO, "Successfully installed diffq package")
                        except Exception as e:
                            self.log(logging.ERROR, f"Failed to install diffq: {e}")
                            self.log(logging.WARNING, "Falling back to mdx model which doesn't require diffq")
                    
                    # Select model based on diffq availability
                    model = "mdx_extra_q" if has_diffq or self._check_diffq_installed() else "mdx"
                    self.log(logging.INFO, f"Using Demucs model: {model}")
                else:
                    # In test environment, use simple model to avoid dependencies
                    model = "mdx"
                
                cmd = [
                    "demucs", 
                    "--two-stems=vocals",
                    "-n", model,  # Use model based on environment
                    "--verbose" # Enable verbose output
                ]
                
                # Add GPU-specific parameters if GPU is available
                if use_gpu:
                    cmd.extend([
                        "--device", "cuda",  # Use CUDA device
                        "--shifts", "2",    # Use 2 shifts for better quality with GPU
                        "--float32"         # Use float32 precision for better GPU utilization
                    ])
                    
                    # Check available GPU memory and adjust segment size accordingly
                    # Note: Demucs doesn't have a batch-size parameter, but uses segment size for memory control
                    try:
                        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
                        
                        # 2025-04-25 -JS - More conservative memory settings to prevent OOM errors
                        # Even with 11GB VRAM, we've seen OOM errors with --no-split
                        if gpu_mem > 16:  # Only use no-split with very high memory GPUs (16GB+)
                            # For very high memory GPUs, we can use no split
                            cmd.append("--no-split")  # Process the entire audio at once
                            # Add overlap for better quality
                            cmd.extend(["--overlap", "0.25"])  # 25% overlap between segments
                        elif gpu_mem > 8:  # For high memory GPUs (8-16GB)
                            # Use larger segments but still segment to avoid OOM
                            cmd.extend(["--segment", "60"])  # Process in 60-second segments
                            cmd.extend(["--overlap", "0.2"])  # 20% overlap between segments
                        elif gpu_mem > 4:  # If more than 4GB VRAM
                            # For medium memory GPUs, use medium segments
                            cmd.extend(["--segment", "30"])  # Process in 30-second segments
                            # Add overlap for better quality with medium-memory GPUs
                            cmd.extend(["--overlap", "0.1"])  # 10% overlap between segments
                        else:  # Limited VRAM
                            # For limited memory GPUs, use smaller segments
                            cmd.extend(["--segment", "10"])  # Process in 10-second segments
                            
                        # Set number of jobs (threads) for processing
                        cmd.extend(["-j", "4"])  # Use 4 threads for better CPU/GPU parallelization
                        
                        self.log(logging.INFO, f"Optimized Demucs memory usage based on {gpu_mem:.1f}GB GPU memory")
                    except Exception as e:
                        self.log(logging.WARNING, f"Could not determine optimal segment size: {str(e)}")
                        cmd.extend(["--segment", "10"])  # Default to safe segment size
                else:
                    # For CPU, use a smaller segment size to avoid memory issues
                    cmd.extend(["--segment", "8"])  # Process in 8-second segments
                
                # Add output directory and input file
                cmd.extend([
                    "-o", temp_dir,
                    input_file
                ])
                
                self.log(logging.DEBUG, f"Running command: {' '.join(cmd)}")
                
                # 2025-04-25 -JS - Allow progress bar output to be displayed
                # Don't capture output in production to show the Demucs progress bar
                # Use text=True to ensure proper encoding of output
                result = subprocess.run(cmd, capture_output=False, text=True)
                
                if result.returncode != 0:
                    self.log(logging.ERROR, f"Demucs failed with return code {result.returncode}")
                    return False
                
                self.log(logging.INFO, "Demucs processing completed successfully")
                
                # Find the vocals file in the output directory
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                
                # Determine model folder name based on the model used
                model_folder = "mdx_extra_q"
                if "-n" in cmd:
                    model_idx = cmd.index("-n") + 1
                    if model_idx < len(cmd):
                        model_folder = cmd[model_idx]
                
                vocals_path = os.path.join(temp_dir, model_folder, base_name, "vocals.wav")
                
                if not os.path.exists(vocals_path):
                    self.log(logging.ERROR, f"Demucs output file not found: {vocals_path}")
                    return False
                
                # 2025-04-24 -JS - Respect the output format setting
                self.log(logging.DEBUG, f"Processing vocals file to output location...")
                
                # Load the vocals file
                vocals_audio = AudioSegment.from_file(vocals_path)
                
                # Export in the correct format based on the output_format setting
                vocals_audio.export(output_file, format=self.output_format)
                
                self.log(logging.INFO, f"Saved vocal file to {os.path.basename(output_file)} in {self.output_format} format")
                return True
                
        except Exception as e:
            self.log(logging.ERROR, f"Error running demucs: {str(e)}")
            return False
    
    def _normalize_audio(self, audio):
        """
        Normalize audio levels.
        
        Args:
            audio: AudioSegment to normalize
            
        Returns:
            AudioSegment: Normalized audio
        """
        return effects.normalize(audio)
    
    def _apply_highpass(self, audio, cutoff, sample_rate):
        """
        Apply high-pass filter.
        
        Args:
            audio: AudioSegment to filter
            cutoff: Filter cutoff frequency in Hz
            sample_rate: Audio sample rate
            
        Returns:
            AudioSegment: Filtered audio
        """
        b, a = scipy.signal.butter(2, cutoff / (0.5 * sample_rate), btype='high')
        y = np.array(audio.get_array_of_samples())
        filtered = scipy.signal.lfilter(b, a, y)
        return audio._spawn(filtered.astype(audio.array_type))
    
    def _apply_lowpass(self, audio, cutoff, sample_rate):
        """
        Apply low-pass filter.
        
        Args:
            audio: AudioSegment to filter
            cutoff: Filter cutoff frequency in Hz
            sample_rate: Audio sample rate
            
        Returns:
            AudioSegment: Filtered audio
        """
        nyquist = 0.5 * sample_rate
        if cutoff >= nyquist:
            self.log(logging.WARNING, f"Low-pass cutoff {cutoff} Hz is >= Nyquist ({nyquist} Hz). Lowering cutoff.")
            cutoff = nyquist - 100  # Leave a small margin
            
        b, a = scipy.signal.butter(2, cutoff / nyquist, btype='low')
        y = np.array(audio.get_array_of_samples())
        filtered = scipy.signal.lfilter(b, a, y)
        return audio._spawn(filtered.astype(audio.array_type))
    
    def _apply_compression(self, audio):
        """
        Apply dynamic range compression with progress reporting.
        
        Args:
            audio: AudioSegment to compress
            
        Returns:
            AudioSegment: Compressed audio
        """
        self.log(logging.INFO, f"Starting compression (threshold: {self.compression_threshold}dB, ratio: {self.compression_ratio})...")
        
        # Get audio parameters
        sample_width = audio.sample_width
        channels = audio.channels
        frame_rate = audio.frame_rate
        
        # Convert to numpy array for processing
        import numpy as np
        samples = np.array(audio.get_array_of_samples())
        
        # Process in chunks to allow progress reporting
        total_samples = len(samples)
        progress_points = []
        
        # Calculate exact sample positions for each 10% increment
        for percentage in range(0, 101, 10):
            progress_points.append((percentage, int(total_samples * percentage / 100)))
        
        # Create a smaller chunk size to ensure we hit all progress points
        chunk_size = max(1, total_samples // 100)  # Use at least 100 chunks for smoother progress
        
        result = np.array([], dtype=samples.dtype)
        last_reported_percentage = -1
        
        # We don't need to report progress percentages anymore
        # The tqdm-style progress bar will show progress visually
        
        # Process each chunk with progress reporting
        for i in range(0, total_samples, chunk_size):
            # Calculate current progress percentage
            current_position = i
            current_percentage = int((current_position / total_samples) * 100)
            
            # Just track progress without printing anything
            # This avoids disrupting any progress bar display
            if current_percentage % 10 == 0 and current_percentage > last_reported_percentage and current_percentage > 0 and current_percentage < 100:
                last_reported_percentage = current_percentage
            
            # Get the current chunk
            chunk = samples[i:i+chunk_size]
            
            # Apply compression to this chunk using pydub's algorithm
            # Convert to float for processing
            chunk_float = chunk.astype(float) / (1 << (8 * sample_width - 1))
            
            # Apply compression
            threshold = 10 ** (self.compression_threshold / 20.0)
            compressed = np.empty_like(chunk_float)
            
            # Apply compression sample by sample
            for j in range(len(chunk_float)):
                sample = chunk_float[j]
                abs_sample = abs(sample)
                if abs_sample > threshold:
                    # Compress only samples above threshold
                    gain_reduction = abs_sample / threshold
                    gain_reduction = gain_reduction ** (1/self.compression_ratio - 1)
                    compressed[j] = sample * gain_reduction
                else:
                    compressed[j] = sample
            
            # Convert back to original data type
            compressed = (compressed * (1 << (8 * sample_width - 1))).astype(samples.dtype)
            
            # Append to result
            result = np.append(result, compressed)
        
        # No need to report 100% completion
        # Let the progress bar show completion visually
        
        # Convert back to AudioSegment
        from pydub import AudioSegment
        return AudioSegment(
            data=result.tobytes(),
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels
        )
    
    def _adjust_volume(self, audio, gain_db):
        """
        Adjust audio volume.
        
        Args:
            audio: AudioSegment to adjust
            gain_db: Gain in dB to apply
            
        Returns:
            AudioSegment: Volume-adjusted audio
        """
        return audio + gain_db
