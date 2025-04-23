#!/usr/bin/env python3
"""
Manual diarization script that alternates speakers based on pauses.
This is useful when automatic diarization fails to distinguish between similar voices.
"""

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import sys
import os
import time
import numpy as np

# Get input filename from command line argument or use default
input_file = sys.argv[1] if len(sys.argv) > 1 else "vocals_clean.mp3"

# Generate output filename based on input filename
output_file = os.path.splitext(input_file)[0] + ".segments"

print(f"Processing file: {input_file}")
print(f"Output will be saved to: {output_file}")

# Start timing
start_time = time.time()
print("Starting manual diarization...")

# Load audio file
print("Loading audio file...")
audio = AudioSegment.from_file(input_file)

# Parameters for silence detection
silence_thresh = -35  # dB
min_silence_len = 1000  # ms
keep_silence = 300  # ms

# Detect non-silent chunks (speech)
print(f"Detecting speech with silence threshold: {silence_thresh}dB, min silence: {min_silence_len}ms")
speech_chunks = detect_nonsilent(
    audio, 
    min_silence_len=min_silence_len, 
    silence_thresh=silence_thresh
)

# Add keep_silence to the chunks manually
print(f"Adding {keep_silence}ms of silence to each chunk")
expanded_chunks = []
for start, end in speech_chunks:
    expanded_start = max(0, start - keep_silence)
    expanded_end = min(len(audio), end + keep_silence)
    expanded_chunks.append((expanded_start, expanded_end))
speech_chunks = expanded_chunks

# Convert to seconds
speech_segments = [(start/1000, end/1000) for start, end in speech_chunks]

print(f"Found {len(speech_segments)} speech segments")

# Group segments that are close together (less than 1 second apart)
merged_segments = []
current_start = None
current_end = None

print("Merging close segments...")
for start, end in speech_segments:
    if current_start is None:
        current_start = start
        current_end = end
    elif start - current_end < 1.0:  # If segments are less than 1 second apart
        current_end = end  # Extend current segment
    else:
        # Save current segment and start a new one
        merged_segments.append((current_start, current_end))
        current_start = start
        current_end = end

# Add the last segment if there is one
if current_start is not None:
    merged_segments.append((current_start, current_end))

print(f"After merging: {len(merged_segments)} speech segments")

# Assign speakers based on timing patterns
segments = []
current_speaker = "SPEAKER_00"  # Start with first speaker

print("Assigning speakers based on timing patterns...")
for i, (start, end) in enumerate(merged_segments):
    # Calculate segment duration
    duration = end - start
    
    # If there's a longer pause (> 2 seconds) before this segment, switch speakers
    if i > 0:
        pause_duration = start - merged_segments[i-1][1]
        if pause_duration > 2.0:
            # Switch speaker
            current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
    
    # For longer segments (> 10 seconds), assume it might contain both speakers
    if duration > 10:
        # Split into two segments
        mid_point = start + (duration / 2)
        segments.append({
            "start": start,
            "end": mid_point,
            "speaker": current_speaker
        })
        # Switch speaker for second half
        current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
        segments.append({
            "start": mid_point,
            "end": end,
            "speaker": current_speaker
        })
    else:
        # Regular segment
        segments.append({
            "start": start,
            "end": end,
            "speaker": current_speaker
        })

# Print timing information
end_time = time.time()
duration = end_time - start_time
print(f"\nDiarization completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"Found {len(segments)} speaker segments")

# Print and save segments
with open(output_file, "w") as f:
    for seg in segments:
        line = f"Speaker {seg['speaker']} from {seg['start']:.2f}s to {seg['end']:.2f}s"
        print(line)
        f.write(line + "\n")

print(f"\nSegments saved to {output_file}")
print(f"You can now use this file with fast-whisper-minimal-segments.py:")
print(f"python fast-whisper-minimal-segments.py {input_file} {output_file}")
