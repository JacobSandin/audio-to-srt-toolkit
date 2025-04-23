#!/usr/bin/env python3
# SRT generator module for creating subtitle files from diarization results
# Handles formatting and writing SRT files with speaker labels
# 2025-04-23 - JS

import os
import sys
import logging
import datetime
from pathlib import Path


class SRTGenerator:
    """
    Class for generating SRT subtitle files from speaker diarization results.
    Handles timestamp formatting and subtitle entry creation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the SRT generator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
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
    
    def format_timestamp(self, seconds):
        """
        Format seconds as SRT timestamp (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted timestamp string
        """
        # Calculate hours, minutes, seconds, and milliseconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_part = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        # Format as HH:MM:SS,mmm
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"
    
    def generate_srt(self, diarization_result, output_file, speaker_format="{speaker}:", include_timestamps=False):
        """
        Generate an SRT file from diarization results.
        
        Args:
            diarization_result: List of diarization segments with start, end, speaker, and text
            output_file: Path to output SRT file
            speaker_format: Format string for speaker labels
            include_timestamps: Whether to include timestamps in the subtitle text
            
        Returns:
            bool: True if SRT generation was successful, False otherwise
        """
        try:
            self.log(logging.INFO, f"Generating SRT file: {output_file}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write each segment as an SRT entry
                for i, segment in enumerate(diarization_result, 1):
                    # Get segment data
                    start_time = segment['start']
                    end_time = segment['end']
                    speaker = segment['speaker']
                    text = segment.get('text', '')
                    
                    # Format speaker ID (extract numeric part if available)
                    speaker_id = speaker.split('_')[-1] if '_' in speaker else speaker
                    
                    # Format timestamps
                    start_timestamp = self.format_timestamp(start_time)
                    end_timestamp = self.format_timestamp(end_time)
                    
                    # Format speaker label
                    speaker_label = speaker_format.format(speaker=speaker, speaker_id=speaker_id)
                    
                    # Build subtitle text
                    subtitle_text = speaker_label
                    
                    # Add timestamp to text if requested
                    if include_timestamps:
                        subtitle_text += f" [{start_timestamp} --> {end_timestamp}]"
                    
                    # Add transcribed text if available
                    if text:
                        subtitle_text += f" {text}"
                    
                    # Write SRT entry
                    f.write(f"{i}\n")  # Index
                    f.write(f"{start_timestamp} --> {end_timestamp}\n")  # Timestamp range
                    f.write(f"{subtitle_text}\n")  # Text
                    f.write("\n")  # Blank line between entries
            
            self.log(logging.INFO, f"Successfully generated SRT file with {len(diarization_result)} entries")
            return True
            
        except Exception as e:
            self.log(logging.ERROR, f"Error generating SRT file: {str(e)}")
            return False
    
    def merge_segments(self, diarization_result, max_gap=1.0, max_duration=10.0):
        """
        Merge consecutive segments from the same speaker if they are close enough.
        
        Args:
            diarization_result: List of diarization segments
            max_gap: Maximum gap in seconds between segments to merge
            max_duration: Maximum duration in seconds for a merged segment
            
        Returns:
            list: Merged diarization segments
        """
        if not diarization_result:
            return []
            
        merged_result = []
        current_segment = diarization_result[0].copy()
        
        for segment in diarization_result[1:]:
            # Check if this segment should be merged with the current one
            same_speaker = segment['speaker'] == current_segment['speaker']
            gap = segment['start'] - current_segment['end']
            merged_duration = segment['end'] - current_segment['start']
            
            if same_speaker and gap <= max_gap and merged_duration <= max_duration:
                # Merge segments
                current_segment['end'] = segment['end']
                
                # Merge text if available
                if 'text' in segment and 'text' in current_segment:
                    if segment['text'] and current_segment['text']:
                        current_segment['text'] += " " + segment['text']
                    elif segment['text']:
                        current_segment['text'] = segment['text']
            else:
                # Add current segment to results and start a new one
                merged_result.append(current_segment)
                current_segment = segment.copy()
        
        # Add the last segment
        merged_result.append(current_segment)
        
        return merged_result
