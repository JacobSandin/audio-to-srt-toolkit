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
        self.highpass_cutoff = self.config.get('highpass_cutoff', 3750)  # Optimal for Swedish dialect isolation - 2025-04-23 -JS
        self.lowpass_cutoff = self.config.get('lowpass_cutoff', 8000)
        self.compression_threshold = self.config.get('compression_threshold', -10.0)
        self.compression_ratio = self.config.get('compression_ratio', 2.0)
        self.default_gain = self.config.get('default_gain', 6)  # +6dB gain
        
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
    
    def convert_to_wav(self, input_file, output_file, bit_depth=24, sample_rate=48000):
        """
        Convert input audio to high-quality WAV format with specified bit depth and sample rate.
        This is the first step in the preprocessing pipeline to ensure consistent quality.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output WAV file
            bit_depth: Bit depth for the WAV file (16, 24, or 32)
            sample_rate: Sample rate in Hz (e.g., 44100, 48000, 96000)
            
        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            self.log(logging.INFO, f"Converting {input_file} to {bit_depth}-bit {sample_rate}Hz WAV")
            
            # Load the input audio
            audio = AudioSegment.from_file(input_file)
            
            # Ensure stereo (2 channels)
            if audio.channels == 1:
                audio = audio.set_channels(2)
            
            # Resample if necessary
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            
            # Handle bit depth correctly
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
            
            # Verify the output file has the correct properties
            converted_audio = AudioSegment.from_file(output_file)
            self.log(logging.DEBUG, f"Converted audio properties: {converted_audio.channels} channels, "
                                    f"{converted_audio.frame_rate}Hz, {converted_audio.sample_width * 8}-bit")
            
            # Save debug file if debug mode is enabled
            self._save_debug_file(converted_audio, input_file, "wav_conversion")
            
            self.log(logging.INFO, f"Conversion completed: {output_file}")
            return True
            
        except Exception as e:
            self.log(logging.ERROR, f"Error converting to WAV: {str(e)}")
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
        self.log(logging.INFO, f"Starting preprocessing of {input_file}")
        
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
        if self.debug_mode and self.debug_dir:
            # In debug mode, save intermediate files in debug directory with consistent naming
            wav_file = os.path.join(self.debug_dir, f"{timestamped_name}_highquality.wav")
            vocals_file = os.path.join(self.debug_dir, f"{timestamped_name}_vocals.wav")
        else:
            # In normal mode, use temporary files that will be cleaned up
            temp_dir = tempfile.mkdtemp(prefix="audio_toolkit_")
            wav_file = os.path.join(temp_dir, f"{timestamped_name}_highquality.wav")
            vocals_file = os.path.join(temp_dir, f"{timestamped_name}_vocals.wav")
        
        try:
            # Step 1: Convert to high-quality WAV
            if not self.convert_to_wav(input_file, wav_file, 
                                      bit_depth=self.config.get('bit_depth', 24),
                                      sample_rate=self.config.get('sample_rate', 48000)):
                self.log(logging.ERROR, "WAV conversion failed")
                return False
            
            # Step 2: Separate vocals using demucs
            if not self.separate_vocals(wav_file, vocals_file):
                self.log(logging.ERROR, "Vocal separation failed")
                return False
            
            # Step 3: Process the separated vocals
            if not self.process_audio(vocals_file, output_file):
                self.log(logging.ERROR, "Audio processing failed")
                return False
            
            self.log(logging.INFO, f"Preprocessing completed successfully: {output_file}")
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
        self.log(logging.INFO, f"Starting vocal separation process on {input_file}")
        self.log(logging.INFO, f"Using Demucs to extract vocals from background elements...")
        
        # Run Demucs for vocal separation
        start_time = time.time()
        success = self._run_demucs(input_file, output_file)
        end_time = time.time()
        
        if success:
            self.log(logging.INFO, f"Vocal separation completed successfully in {end_time - start_time:.2f} seconds")
            
            # Get info about the separated vocals file
            try:
                vocals_audio = AudioSegment.from_file(output_file)
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
        
        Args:
            audio: AudioSegment to save
            input_file: Original input file path (used for naming)
            step_name: Name of the processing step
            
        Returns:
            None
        """
        if not self.debug_mode or not self.debug_dir:
            return
            
        try:
            # Get base name of input file without extension
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            
            # Check if the input_file already contains a timestamp
            import re
            timestamp_pattern = re.compile(r'\d{8}_\d{6}')
            
            # If no timestamp in the base_name, add one at the beginning
            if not timestamp_pattern.search(base_name):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = f"{timestamp}_{base_name}"
            
            # Create debug filename with step name but no additional timestamp
            debug_filename = f"{base_name}_{step_name}.wav"
            debug_path = os.path.join(self.debug_dir, debug_filename)
            
            # Export debug file in WAV format for better quality
            audio.export(debug_path, format="wav")  # 2025-04-23 - JS
            self.log(logging.DEBUG, f"Saved debug file for {step_name}: {debug_path}")
            
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
        
        # If no timestamp in the base_name, add one at the beginning for output_file
        if not timestamp_pattern.search(base_name) and not timestamp_pattern.search(os.path.basename(output_file)):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # Update output_file with timestamped name if it doesn't already have a timestamp
            output_dir = os.path.dirname(output_file)
            output_filename = os.path.basename(output_file)
            output_base = os.path.splitext(output_filename)[0]
            output_ext = os.path.splitext(output_filename)[1]
            if not timestamp_pattern.search(output_base):
                output_file = os.path.join(output_dir, f"{timestamp}_{output_base}{output_ext}")
                self.log(logging.DEBUG, f"Updated output file path with timestamp: {output_file}")
        
        try:
            self.log(logging.INFO, f"Processing audio: {input_file}")
            
            # Load audio
            self.log(logging.INFO, f"Loading audio file: {input_file}")
            audio = AudioSegment.from_file(input_file)
            sample_rate = audio.frame_rate
            self.log(logging.INFO, f"Audio loaded: {len(audio)/1000:.2f} seconds, {sample_rate}Hz sample rate")
            
            # Apply processing steps and save debug files for each step
            self.log(logging.INFO, f"Applying high-pass filter (cutoff: {self.highpass_cutoff}Hz)...")
            audio = self.apply_highpass(audio, cutoff=self.highpass_cutoff, sample_rate=sample_rate)
            self._save_debug_file(audio, input_file, "highpass")
            self.log(logging.INFO, f"High-pass filter applied successfully")
            
            self.log(logging.INFO, f"Applying low-pass filter (cutoff: {self.lowpass_cutoff}Hz)...")
            audio = self.apply_lowpass(audio, cutoff=self.lowpass_cutoff, sample_rate=sample_rate)
            self._save_debug_file(audio, input_file, "lowpass")
            self.log(logging.INFO, f"Low-pass filter applied successfully")
            
            self.log(logging.INFO, f"Applying compression (threshold: {self.compression_threshold}dB, ratio: {self.compression_ratio})...")
            audio = self.apply_compression(audio)
            self._save_debug_file(audio, input_file, "compression")
            self.log(logging.INFO, f"Compression applied successfully")
            
            self.log(logging.INFO, f"Normalizing audio levels...")
            audio = self.normalize(audio)
            self._save_debug_file(audio, input_file, "normalize")
            self.log(logging.INFO, f"Normalization completed successfully")
            
            self.log(logging.INFO, f"Adjusting volume (gain: {self.default_gain}dB)...")
            audio = self.adjust_volume(audio, gain_db=self.default_gain)
            self._save_debug_file(audio, input_file, "volume")
            self.log(logging.INFO, f"Volume adjustment completed successfully")
            
            # Export processed audio in WAV format for better quality
            audio.export(output_file, format="wav")
            self.log(logging.INFO, f"Audio processing completed: {output_file}")
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
            self.log(logging.INFO, "Initializing Demucs for vocal separation")
            
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
                cmd = [
                    "demucs", 
                    "--two-stems=vocals",
                    "--verbose", # Enable verbose output
                    "-o", temp_dir,
                    input_file
                ]
                
                self.log(logging.DEBUG, f"Running command: {' '.join(cmd)}")
                
                # Start process with real-time output monitoring
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout for unified output
                    universal_newlines=True,   # Return strings instead of bytes
                    bufsize=1                 # Line buffered
                )
                
                # Process output in real-time to provide progress updates
                last_progress = 0
                progress_line = ""
                model_count = 1  # Default to 1 model
                current_model = 0
                
                # Look for progress indicators in the output
                for line in process.stdout:
                    line = line.strip()
                    self.log(logging.DEBUG, line)  # Log all output at DEBUG level for troubleshooting
                    
                    # Check if this is a bag of models and update the count
                    if "bag of" in line and "models" in line:
                        try:
                            model_count = int(line.split("bag of")[1].split("models")[0].strip())
                            self.log(logging.INFO, f"Demucs will process using {model_count} models")
                        except:
                            pass
                    
                    # Track which model is currently processing
                    if "Model #" in line:
                        try:
                            current_model = int(line.split("Model #")[1].split(":")[0].strip())
                            self.log(logging.INFO, f"Processing with model {current_model} of {model_count}")
                        except:
                            pass
                    
                    # Handle progress information for the new htdemucs model
                    if "|" in line and "%" in line and ("it/s" in line or "s/it" in line):
                        try:
                            # New format has progress like: 100%|██████████| 30/30 [00:21<00:00,  1.40it/s]
                            progress_part = line.split("|")[0].strip()
                            if "%" in progress_part:
                                progress = int(progress_part.split("%")[0].strip())
                                
                                # Calculate overall progress considering multiple models
                                if model_count > 1 and current_model > 0:
                                    overall_progress = int((current_model - 1) * 100 / model_count + progress / model_count)
                                else:
                                    overall_progress = progress
                                
                                # Only log at significant milestones (0%, 25%, 50%, 75%, 100%)
                                milestone_reached = False
                                for milestone in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                                    if overall_progress >= milestone and last_progress < milestone:
                                        milestone_reached = True
                                        break
                                
                                if milestone_reached:
                                    self.log(logging.INFO, f"Demucs progress: {overall_progress}%")
                                    last_progress = overall_progress
                        except Exception as e:
                            # If we can't parse the progress, just continue
                            pass
                    
                    # Handle the old progress format as well
                    elif "Progress" in line:
                        try:
                            progress = int(line.split('%')[0].split()[-1])
                            
                            # Only log at significant milestones (0%, 25%, 50%, 75%, 100%)
                            if progress in [0, 25, 50, 75, 100] and progress > last_progress:
                                self.log(logging.INFO, f"Demucs progress: {progress}%")
                                last_progress = progress
                        except:
                            pass
                    
                    # Log important status messages at INFO level
                    elif any(keyword in line for keyword in ["Separated", "Saving", "Done", "Model", "Using", "Loading"]):
                        self.log(logging.INFO, line)
                    
                    # Log everything else at DEBUG level
                    else:
                        self.log(logging.DEBUG, line)
                
                # Wait for process to complete
                process.wait()
                
                if process.returncode != 0:
                    self.log(logging.ERROR, f"Demucs failed with return code {process.returncode}")
                    return False
                
                self.log(logging.INFO, "Demucs processing completed successfully")
                
                # Find the vocals file in the output directory
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                vocals_path = os.path.join(temp_dir, "htdemucs", base_name, "vocals.wav")
                
                if not os.path.exists(vocals_path):
                    self.log(logging.ERROR, f"Demucs output file not found: {vocals_path}")
                    return False
                
                # Copy the vocals file to the output location
                self.log(logging.INFO, f"Copying vocals file to output location...")
                shutil.copy(vocals_path, output_file)
                self.log(logging.INFO, f"Vocals extracted and saved to {output_file}")
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
        chunk_size = len(samples) // 10  # Process in 10 chunks for 10% progress updates
        if chunk_size == 0:
            chunk_size = len(samples)
        
        result = np.array([], dtype=samples.dtype)
        
        # Report 0% at start
        self.log(logging.INFO, "Compression progress: 0%")
        
        # Process each chunk with progress reporting
        for i in range(0, len(samples), chunk_size):
            # Calculate progress percentage
            progress = min(100, int((i / len(samples)) * 100))
            
            # Report at every 10% increment (10%, 20%, 30%, etc.)
            # Skip 0% (already reported) and 100% (will report at end)
            if progress % 10 == 0 and progress > 0 and progress < 100:
                self.log(logging.INFO, f"Compression progress: {progress}%")
            
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
        
        # Report 100% completion
        self.log(logging.INFO, "Compression progress: 100%")
        
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
