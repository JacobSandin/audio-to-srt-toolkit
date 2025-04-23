#!/usr/bin/env python3
"""
Convert audio to a format that works better with pyannote.audio 3.1.1
- Converts to WAV format
- Sets sample rate to 16kHz
- Converts to mono
- Normalizes audio levels
"""

import os
import sys
import subprocess
import time

def convert_audio(input_file):
    """Convert audio to a format that works better with pyannote.audio"""
    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_diarize.wav"
    
    print(f"Converting {input_file} to {output_file}")
    print("- Converting to WAV format")
    print("- Setting sample rate to 16kHz")
    print("- Converting to mono")
    print("- Normalizing audio levels")
    
    # Use ffmpeg to convert the file
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-ac", "1",               # Convert to mono
        "-ar", "16000",           # Set sample rate to 16kHz
        "-sample_fmt", "s16",     # 16-bit samples
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",  # Normalize audio levels
        output_file
    ]
    
    # Run the command
    start_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()
    
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Get input file from command line
    if len(sys.argv) < 2:
        print("Usage: python convert_for_diarization.py <input_audio_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = convert_audio(input_file)
    
    print("\nNext steps:")
    print(f"1. Run diarization: python diarization.py {output_file}")
