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
        self.highpass_cutoff = self.config.get('highpass_cutoff', 150)
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
        
        # Step 1: Convert to high-quality WAV
        wav_file = os.path.splitext(output_file)[0] + "_highquality.wav"
        if not self.convert_to_wav(input_file, wav_file, 
                                  bit_depth=self.config.get('bit_depth', 24),
                                  sample_rate=self.config.get('sample_rate', 48000)):
            self.log(logging.ERROR, "WAV conversion failed")
            return False
        
        # Step 2: Separate vocals using demucs
        vocals_file = os.path.splitext(output_file)[0] + "_vocals.wav"
        if not self.separate_vocals(wav_file, vocals_file):
            self.log(logging.ERROR, "Vocal separation failed")
            return False
        
        # Step 3: Process the separated vocals
        if not self.process_audio(vocals_file, output_file):
            self.log(logging.ERROR, "Audio processing failed")
            return False
        
        self.log(logging.INFO, f"Preprocessing completed successfully: {output_file}")
        return True
    
    def separate_vocals(self, input_file, output_file):
        """
        Separate vocals from background using demucs.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            
        Returns:
            bool: True if separation was successful, False otherwise
        """
        self.log(logging.INFO, f"Separating vocals from {input_file}")
        success = self._run_demucs(input_file, output_file)
        
        # Save debug file for vocals if successful
        if success and self.debug_mode and self.debug_dir:
            try:
                vocals_audio = AudioSegment.from_file(output_file)
                self._save_debug_file(vocals_audio, input_file, "vocals")
            except Exception as e:
                self.log(logging.WARNING, f"Failed to save debug file for vocals: {str(e)}")
                
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
            
            # Create timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            
            # Create debug filename
            debug_filename = f"{base_name}.{step_name}.{timestamp}.wav"
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
        try:
            self.log(logging.INFO, f"Processing audio: {input_file}")
            
            # Load audio
            audio = AudioSegment.from_file(input_file)
            sample_rate = audio.frame_rate
            
            # Apply processing steps and save debug files for each step
            audio = self.apply_highpass(audio, cutoff=self.highpass_cutoff, sample_rate=sample_rate)
            self._save_debug_file(audio, input_file, "highpass")
            
            audio = self.apply_lowpass(audio, cutoff=self.lowpass_cutoff, sample_rate=sample_rate)
            self._save_debug_file(audio, input_file, "lowpass")
            
            audio = self.apply_compression(audio)
            self._save_debug_file(audio, input_file, "compression")
            
            audio = self.normalize(audio)
            self._save_debug_file(audio, input_file, "normalize")
            
            audio = self.adjust_volume(audio, gain_db=self.default_gain)
            self._save_debug_file(audio, input_file, "volume")
            
            # Export processed audio
            audio.export(output_file, format="mp3", bitrate="192k")
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
            self.log(logging.INFO, "Running demucs for vocal separation")
            
            # Create a temporary directory for demucs output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run demucs command
                cmd = [
                    "demucs", 
                    "--two-stems=vocals", 
                    "-o", temp_dir,
                    input_file
                ]
                
                self.log(logging.DEBUG, f"Running command: {' '.join(cmd)}")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    self.log(logging.ERROR, f"Demucs failed: {stderr.decode()}")
                    return False
                
                # Find the vocals file in the output directory
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                vocals_path = os.path.join(temp_dir, "htdemucs", base_name, "vocals.wav")
                
                if not os.path.exists(vocals_path):
                    self.log(logging.ERROR, f"Demucs output file not found: {vocals_path}")
                    return False
                
                # Copy the vocals file to the output location
                shutil.copy(vocals_path, output_file)
                self.log(logging.INFO, f"Vocals extracted to {output_file}")
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
        Apply dynamic range compression.
        
        Args:
            audio: AudioSegment to compress
            
        Returns:
            AudioSegment: Compressed audio
        """
        return effects.compress_dynamic_range(
            audio, 
            threshold=self.compression_threshold,
            ratio=self.compression_ratio
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
