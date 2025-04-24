"""
Transcription module for the audio-to-srt toolkit.
Uses faster-whisper to transcribe audio segments and continuously update SRT files.

2025-04-24 -JS
"""

import os
import logging
import tempfile
import torch
import warnings
import contextlib
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Suppress ResourceWarnings
warnings.filterwarnings("ignore", category=ResourceWarning)

class WhisperTranscriber:
    """
    Class for transcribing audio segments using faster-whisper.
    Includes continuous SRT file updating.
    
    2025-04-24 -JS
    """
    
    def __init__(self, config=None):
        """
        Initialize the transcriber with configuration.
        
        Args:
            config: Dictionary with configuration options
        """
        self.config = config or {}
        self.model = None
        self.model_name = self.config.get('model_name', 'large-v3')
        self.language = self.config.get('language', 'sv')  # Default to Swedish
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = self.config.get('compute_type', 'float16')
        self.timestamp_offset = self.config.get('timestamp_offset', 0.07)  # Small offset for better alignment
        
        # Segment padding for better transcription of short expressions
        # 2025-04-24 -JS
        self.pre_padding = self.config.get('pre_padding', 0.1)  # Seconds to add before segment
        self.post_padding = self.config.get('post_padding', 0.1)  # Seconds to add after segment
        
        # Confidence threshold for transcription
        # 2025-04-24 -JS
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)  # Minimum confidence threshold
        
        # Minimum segment duration
        # 2025-04-24 -JS
        self.min_segment_duration = self.config.get('min_segment_duration', 0.3)  # Minimum segment duration in seconds
        
        # Whether to include speaker labels in SRT output
        # 2025-04-24 -JS
        self.include_speaker = self.config.get('include_speaker', True)  # Include speaker labels by default
        
        # Whether to remove empty segments from SRT output
        # 2025-04-24 -JS
        self.remove_empty_segments = self.config.get('remove_empty_segments', False)  # Don't remove empty segments by default
        
        # Placeholder text for empty segments
        # 2025-04-24 -JS
        self.empty_placeholder = self.config.get('empty_placeholder', '[UNRECOGNIZABLE]')  # Default is [UNRECOGNIZABLE]
        
        # Hallucination patterns to filter out
        # 2025-04-24 -JS
        self.hallucination_patterns = [
            # Common hallucinations from training data
            "textning.nu",
            "undertexter från",
            "amara.org",
            "tack till elever",
            "värmlands universi",
            "gemenskapen",
            "Stina Hedin",
            "btistudio",
            "&lt;i&gt;",
            
            # Swedish-specific hallucinations for short segments
            "uppdateringar",
            "uppehåll",
            "uppdatering",
            "undertextning",
            "undertext",
            "textning"
        ]
        
        # Short segment hallucination detection
        # For very short segments, we need stricter filtering
        # 2025-04-24 -JS
        self.short_segment_hallucination_patterns = [
            # Common hallucinations for very short segments
            "ja",  # Only when it's not clearly audible
            "nej",  # Only when it's not clearly audible
            "mm",
            "hm",
            "eh",
            "öh"
        ]
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # SRT file handle for continuous updates
        self.srt_file = None
        self.srt_counter = 1
    
    def log(self, level, *messages, **kwargs):
        """
        Log a message at the specified level.
        
        Args:
            level: Logging level
            *messages: Messages to log
            **kwargs: Additional logging parameters
        """
        message = " ".join(str(msg) for msg in messages)
        self.logger.log(level, message, **kwargs)
    
    def load_model(self):
        """
        Load the whisper model if not already loaded.
        
        Returns:
            bool: True if model loaded successfully
        """
        if self.model is not None:
            return True
            
        try:
            self.log(logging.INFO, f"Loading Whisper model: {self.model_name}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root="cache"
            )
            self.log(logging.INFO, f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            self.log(logging.ERROR, f"Error loading Whisper model: {str(e)}")
            return False
    
    def is_hallucinated_text(self, text, segment_duration=None):
        """
        Check if text matches common hallucination patterns.
        Uses stricter filtering for very short segments.
        
        Args:
            text: Text to check
            segment_duration: Duration of the segment in seconds (optional)
            
        Returns:
            bool: True if text is likely hallucinated
        
        2025-04-24 -JS
        """
        if not text or len(text.strip()) == 0:
            return False
            
        text_lower = text.lower()
        
        # Check against standard hallucination patterns
        for pattern in self.hallucination_patterns:
            if pattern.lower() in text_lower:
                self.log(logging.DEBUG, f"Detected hallucination: '{text}' matches pattern '{pattern}'")
                return True
        
        # For very short segments, apply stricter filtering but only for extremely short segments
        # 2025-04-24 -JS - Made less aggressive to avoid filtering important content
        if segment_duration and segment_duration < 0.2:  # Reduced from 0.3s to 0.2s
            # For extremely short segments, check if the text is suspiciously long
            if len(text) > segment_duration * 30:  # Increased from 20 to 30 chars per second
                self.log(logging.DEBUG, f"Detected hallucination: '{text}' too long for {segment_duration:.2f}s segment")
                return True
                
            # Check against short segment hallucination patterns
            # Only apply for very short segments with simple utterances
            if len(text) < 5:  # Only apply to very short text
                for pattern in self.short_segment_hallucination_patterns:
                    if pattern.lower() == text_lower or f"{pattern.lower()}." == text_lower:
                        self.log(logging.DEBUG, f"Detected short segment hallucination: '{text}' matches pattern '{pattern}'")
                        return True
        
        return False
    
    def format_timestamp(self, seconds):
        """
        Format seconds as SRT timestamp (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted timestamp
        """
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @contextlib.contextmanager
    def open_audio(self, audio_file):
        """
        Context manager for safely opening audio files.
        
        Args:
            audio_file: Path to the audio file
            
        Yields:
            AudioSegment: The loaded audio
        """
        try:
            audio = AudioSegment.from_file(audio_file)
            yield audio
        finally:
            # No explicit cleanup needed for AudioSegment
            pass
    
    def transcribe_segment(self, audio, segment):
        """
        Transcribe a specific segment of an audio file.
        
        Args:
            audio: AudioSegment object
            segment: Dictionary with start and end times
            
        Returns:
            str: Transcribed text
        """
        try:
            # Skip segments that are too short
            if segment['end'] - segment['start'] < 0.1:
                return ""
            
            # Apply padding to segment boundaries for better transcription
            # 2025-04-24 -JS
            audio_duration_ms = len(audio)
            padded_start_ms = max(0, int((segment['start'] - self.pre_padding) * 1000))
            padded_end_ms = min(audio_duration_ms, int((segment['end'] + self.post_padding) * 1000))
            
            # Extract segment with padding
            segment_audio = audio[padded_start_ms:padded_end_ms]
            
            # Log the padding applied
            actual_pre_padding = (int(segment['start'] * 1000) - padded_start_ms) / 1000
            actual_post_padding = (padded_end_ms - int(segment['end'] * 1000)) / 1000
            
            # Only log if there's significant padding
            if actual_pre_padding > 0.01 or actual_post_padding > 0.01:
                self.log(logging.DEBUG, f"Applied padding: pre={actual_pre_padding:.2f}s, post={actual_post_padding:.2f}s")
            
            # Use a context manager for the temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Export to the temporary file
                segment_audio.export(temp_filename, format="wav")
                
                # Make sure model is loaded
                if not self.load_model():
                    return ""
                
                # Transcribe with confidence scores
                # 2025-04-24 -JS
                transcription, info = self.model.transcribe(
                    temp_filename, 
                    language=self.language,
                    beam_size=5,
                    word_timestamps=True,
                    condition_on_previous_text=False,  # Don't condition on previous text to avoid bias
                    vad_filter=True,  # Filter out non-speech
                    vad_parameters={"threshold": 0.01}  # Even lower VAD threshold to catch more speech
                )
                
                # Get the text
                transcription_list = list(transcription)
                
                # Process each transcription segment
                full_text = []
                for trans_seg in transcription_list:
                    text = trans_seg.text.strip()
                    
                    # Skip completely empty segments
                    if not text:
                        continue
                    
                    # Calculate segment duration for better hallucination detection
                    segment_duration = (segment['end'] - segment['start'])
                    
                    # Check confidence score if available, but with a lower threshold for short segments
                    # 2025-04-24 -JS
                    if hasattr(trans_seg, 'avg_logprob'):
                        # Convert log probability to confidence score (0-1)
                        confidence = min(1.0, max(0.0, 1.0 + trans_seg.avg_logprob))
                        # Use a lower threshold for short segments to catch more content
                        effective_threshold = self.confidence_threshold
                        if segment_duration < 2.0:  # For short segments
                            effective_threshold = max(0.2, self.confidence_threshold - 0.2)  # Lower threshold by 0.2 but not below 0.2
                            
                        # Mark low confidence text with [L] prefix instead of skipping
                        # 2025-04-24 -JS
                        if confidence < effective_threshold:
                            self.log(logging.DEBUG, f"Marking low confidence text: '{text}' (confidence: {confidence:.2f} < {effective_threshold:.2f})")
                            text = f"[L] {text}"
                    
                    # Only check for hallucinations if the text is suspiciously long or matches known patterns
                    # 2025-04-24 -JS
                    if len(text) > segment_duration * 30 or any(pattern.lower() in text.lower() for pattern in self.hallucination_patterns):
                        if self.is_hallucinated_text(text, segment_duration):
                            self.log(logging.DEBUG, f"Skipping hallucinated text: '{text}' in {segment_duration:.2f}s segment")
                            continue
                    
                    full_text.append(text)
                
                return " ".join(full_text)
            finally:
                # Always clean up the temporary file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
        
        except Exception as e:
            self.log(logging.ERROR, f"Error transcribing segment: {str(e)}")
            return ""
    
    def open_srt_file(self, output_file):
        """
        Open an SRT file for continuous writing.
        
        Args:
            output_file: Path to the SRT file
            
        Returns:
            bool: True if file opened successfully
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Open file for writing
            self.srt_file = open(output_file, 'w', encoding='utf-8')
            self.srt_counter = 1
            return True
        except Exception as e:
            self.log(logging.ERROR, f"Error opening SRT file: {str(e)}")
            return False
    
    def close_srt_file(self):
        """
        Close the SRT file if open.
        """
        if self.srt_file:
            self.srt_file.close()
            self.srt_file = None
    
    def write_srt_entry(self, segment):
        """
        Write a single entry to the SRT file.
        
        Args:
            segment: Dictionary with start, end, speaker, and text
            
        Returns:
            bool: True if entry written successfully
        """
        if not self.srt_file:
            return False
            
        try:
            # Get segment data
            start_sec = segment['start'] + self.timestamp_offset  # Add offset for better alignment
            end_sec = segment['end']
            speaker = segment['speaker']
            text = segment.get('text', '')
            
            # Use placeholder for empty segments if specified
            # 2025-04-24 -JS
            if not text.strip() and self.empty_placeholder:
                text = self.empty_placeholder
            
            # Format timestamps
            start = self.format_timestamp(start_sec)
            end = self.format_timestamp(end_sec)
            
            # Build subtitle text based on whether to include speaker labels
            # 2025-04-24 -JS
            if self.include_speaker:
                # Format speaker label
                speaker_label = f"{speaker}:"
                # Build subtitle text with speaker label
                subtitle_text = f"{speaker_label} {text}" if text else speaker_label
            else:
                # Build subtitle text without speaker label
                subtitle_text = text
            
            # Write SRT entry
            self.srt_file.write(f"{self.srt_counter}\n")  # Index
            self.srt_file.write(f"{start} --> {end}\n")  # Timestamp range
            self.srt_file.write(f"{subtitle_text}\n")  # Text
            self.srt_file.write("\n")  # Blank line between entries
            
            # Flush to ensure immediate write
            self.srt_file.flush()
            
            # Increment counter
            self.srt_counter += 1
            
            return True
        except Exception as e:
            self.log(logging.ERROR, f"Error writing SRT entry: {str(e)}")
            return False
    
    def transcribe_segments_to_srt(self, audio_file, segments, output_file):
        """
        Transcribe segments and write directly to an SRT file.
        
        Args:
            audio_file: Path to the audio file
            segments: List of segment dictionaries with start and end times
            output_file: Path to the output SRT file
            
        Returns:
            tuple: (success, processed_count, filtered_count)
                - success: True if transcription was successful
                - processed_count: Number of segments successfully processed
                - filtered_count: Number of segments filtered out
        """
        self.log(logging.INFO, f"Transcribing {len(segments)} segments to {output_file}")
        
        # Open SRT file
        if not self.open_srt_file(output_file):
            return False, 0, 0
        
        try:
            # Make sure model is loaded
            if not self.load_model():
                return False, 0, 0
            
            # Open audio file once for all segments
            with self.open_audio(audio_file) as audio:
                # Process segments in order of start time
                sorted_segments = sorted(segments, key=lambda x: x['start'])
                
                # Count filtered segments and processed segments
                filtered_count = 0
                processed_count = 0
                
                # Process each segment
                for i, segment in enumerate(sorted_segments):
                    # Log progress every 10 segments to reduce noise
                    if i % 10 == 0:
                        self.log(logging.INFO, f"Processing segment {i+1}/{len(sorted_segments)}")
                    
                    # Calculate segment duration
                    segment_duration = segment['end'] - segment['start']
                    
                    # First check if the segment meets the minimum duration requirement
                    # 2025-04-24 -JS - Apply filtering before transcription and padding
                    if (self.min_segment_duration > 0) and segment_duration < self.min_segment_duration:
                        self.log(logging.DEBUG, f"Skipping segment {i+1} (duration: {segment_duration:.2f}s) below minimum duration {self.min_segment_duration}s")
                        filtered_count += 1
                        continue
                    
                    # Skip extremely short segments that are likely to be meaningless
                    # 2025-04-24 -JS - This is a separate check from min_segment_duration
                    if segment_duration < 0.15 and self.remove_empty_segments:
                        self.log(logging.DEBUG, f"Skipping very short segment {i+1} (duration: {segment_duration:.2f}s) due to --srt-remove-empty")
                        filtered_count += 1
                        continue
                    
                    # Only transcribe segments that passed the duration checks
                    # 2025-04-24 -JS
                    text = self.transcribe_segment(audio, segment)
                    
                    # Update segment with transcribed text
                    segment['text'] = text if text else ""
                    
                    # Skip completely empty segments after transcription if remove_empty_segments is True
                    # 2025-04-24 -JS - Only check for empty text if remove_empty_segments is enabled
                    # We've already filtered by min_segment_duration before transcription
                    if self.remove_empty_segments and not segment['text'].strip() and not self.empty_placeholder:
                        self.log(logging.DEBUG, f"Skipping segment {i+1} with empty transcription due to --srt-remove-empty")
                        filtered_count += 1
                        continue
                    
                    # Log the transcription for debugging
                    self.log(logging.DEBUG, f"Segment {i+1} ({segment_duration:.2f}s): '{segment['text']}'")
                    
                    # Write to SRT file
                    self.write_srt_entry(segment)
                    processed_count += 1
                
                # Log how many segments were filtered out
                if filtered_count > 0:
                    self.log(logging.INFO, f"Filtered out {filtered_count} segments out of {len(sorted_segments)}")
                
                self.log(logging.INFO, f"Successfully processed {processed_count} segments out of {len(sorted_segments)}")
            
            return True, processed_count, filtered_count
        except Exception as e:
            self.log(logging.ERROR, f"Error during transcription: {str(e)}")
            return False, 0, 0
        finally:
            # Always close the SRT file
            self.close_srt_file()
