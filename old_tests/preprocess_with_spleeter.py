import sys
import os
import subprocess
import shutil
import numpy as np
from pydub import AudioSegment, effects

def apply_noise_gate(audio_file, output_file, threshold_db=-25, min_silence_ms=300):
    """
    Apply a noise gate to remove breathing and quiet sounds
    """
    print(f"Applying noise gate to remove breathing sounds...")
    audio = AudioSegment.from_file(audio_file)
    
    # Split audio into chunks
    chunks = []
    silence_thresh = audio.dBFS + threshold_db
    
    # Process in 1-second chunks for efficiency
    chunk_size = 1000  # 1 second
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        # If chunk is louder than threshold, keep it
        if chunk.dBFS > silence_thresh:
            chunks.append(chunk)
    
    # If no chunks passed the threshold, keep the loudest part
    if not chunks and len(audio) > 0:
        loudest_start = 0
        loudest_db = -100
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if chunk.dBFS > chunk.dBFS:
                loudest_db = chunk.dBFS
                loudest_start = i
        chunks.append(audio[loudest_start:loudest_start+chunk_size])
    
    # Combine chunks with crossfade to avoid clicks
    if chunks:
        result = chunks[0]
        for chunk in chunks[1:]:
            result = result.append(chunk, crossfade=10)
    else:
        result = audio
    
    # Normalize the result
    result = effects.normalize(result)
    
    # Export the result
    result.export(output_file, format="mp3", bitrate="192k")
    return True

def separate_with_spleeter(input_file, output_file, remove_breathing=True):
    """
    Use spleeter to separate vocals from background noise
    """
    print(f"Processing {input_file} with spleeter...")
    
    # Create a temporary directory for spleeter output
    temp_dir = "spleeter_output"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Run spleeter to separate vocals
    cmd = ["spleeter", "separate", "-p", "spleeter:2stems", "-o", temp_dir, input_file]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Find the vocals track
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    vocals_path = os.path.join(temp_dir, base_name, "vocals.wav")
    
    if not os.path.exists(vocals_path):
        print(f"Could not find vocals at {vocals_path}")
        print("Checking for other locations...")
        # Try to find the vocals file
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if "vocals" in file.lower() and file.endswith((".wav", ".mp3")):
                    vocals_path = os.path.join(root, file)
                    print(f"Found vocals at {vocals_path}")
                    break
    
    if not os.path.exists(vocals_path):
        print("Could not find separated vocals. Check spleeter installation.")
        shutil.rmtree(temp_dir)
        return False
    
    # Apply noise gate to remove breathing if requested
    if remove_breathing:
        temp_output = "temp_vocals_no_breathing.mp3"
        apply_noise_gate(vocals_path, temp_output)
        shutil.copy(temp_output, output_file)
        os.remove(temp_output)
        print(f"Vocals with breathing removed saved to {output_file}")
    else:
        # Just copy the vocals to the output location
        shutil.copy(vocals_path, output_file)
        print(f"Vocals saved to {output_file}")
    
    # Clean up
    shutil.rmtree(temp_dir)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_with_spleeter.py input.mp3 output.mp3 [--keep-breathing]")
        print("  --keep-breathing: Optional flag to preserve breathing sounds")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if user wants to keep breathing sounds
    remove_breathing = True
    if len(sys.argv) > 3 and sys.argv[3] == "--keep-breathing":
        remove_breathing = False
        print("Breathing sounds will be preserved")
    else:
        print("Breathing sounds will be removed")
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        sys.exit(1)
    
    success = separate_with_spleeter(input_file, output_file, remove_breathing=remove_breathing)
    if success:
        print("\nProcessing complete!")
        print("\nNext steps:")
        print(f"1. Run diarization: python diarization.py {output_file}")
        print(f"2. Run transcription: python fast-whisper-minimal-segments.py {input_file} {os.path.splitext(output_file)[0]}.segments")
    else:
        print("Processing failed")
