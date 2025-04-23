import sys
import os
import subprocess
import shutil
import numpy as np
from pydub import AudioSegment, effects

def apply_noise_gate(audio_file, output_file, threshold_db=-25, min_silence_ms=300):
    """
    Apply a noise gate to remove breathing sounds - optimized for speed
    """
    print(f"Applying noise gate to remove breathing sounds...")
    audio = AudioSegment.from_file(audio_file)
    
    # Calculate threshold relative to this audio file
    silence_thresh = audio.dBFS + threshold_db
    print(f"Audio dBFS: {audio.dBFS}, threshold: {silence_thresh}")
    
    # Use a much larger chunk size for faster processing
    chunk_size = 5000  # 5 seconds
    
    # Estimate number of chunks
    estimated_chunks = len(audio) // chunk_size + 1
    chunks = []
    
    print(f"Processing {len(audio)/1000:.1f} seconds of audio in {chunk_size/1000:.1f}-second chunks...")
    
    # First pass - just identify which chunks to keep (faster)
    keep_indices = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if chunk.dBFS > silence_thresh:
            keep_indices.append(i)
    
    # If nothing was above threshold, keep the loudest chunk
    if not keep_indices and len(audio) > 0:
        print("No chunks above threshold, keeping loudest section...")
        loudest_start = 0
        loudest_db = -100
        # Sample fewer points for speed
        for i in range(0, len(audio), chunk_size * 2):
            chunk = audio[i:i+chunk_size]
            if chunk.dBFS > loudest_db:
                loudest_db = chunk.dBFS
                loudest_start = i
        keep_indices.append(loudest_start)
    
    # Second pass - extract only the chunks we're keeping
    print(f"Keeping {len(keep_indices)} chunks out of {estimated_chunks}")
    for i in keep_indices:
        chunks.append(audio[i:i+chunk_size])
    
    # Combine chunks with crossfade to avoid clicks
    if chunks:
        print("Combining chunks...")
        result = chunks[0]
        for chunk in chunks[1:]:
            result = result.append(chunk, crossfade=20)
    else:
        result = audio
    
    # Normalize the result
    print("Normalizing audio...")
    result = effects.normalize(result)
    
    # Export the result
    print(f"Exporting to {output_file}...")
    result.export(output_file, format="mp3", bitrate="192k")
    print("Noise gate processing complete!")
    return True

def separate_with_demucs(input_file, output_file, remove_breathing=True):
    """
    Use demucs to separate vocals from background noise
    """
    print(f"Processing {input_file} with demucs...")
    
    # Create a temporary directory for demucs output
    temp_dir = "demucs_output"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Run demucs to separate vocals
    cmd = ["demucs", "--two-stems=vocals", "-o", temp_dir, input_file]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Find the vocals track
    model_name = "htdemucs"  # Default model name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    vocals_path = os.path.join(temp_dir, model_name, base_name, "vocals.wav")
    
    if not os.path.exists(vocals_path):
        print(f"Could not find vocals at {vocals_path}")
        print("Checking for other model names...")
        # Try to find the vocals file
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file == "vocals.wav":
                    vocals_path = os.path.join(root, file)
                    print(f"Found vocals at {vocals_path}")
                    break
    
    if not os.path.exists(vocals_path):
        print("Could not find separated vocals. Check demucs installation.")
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
        print("Usage: python preprocess_with_demucs.py input.mp3 output.mp3 [--keep-breathing]")
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
    
    success = separate_with_demucs(input_file, output_file, remove_breathing=remove_breathing)
    if success:
        print("\nProcessing complete!")
        print("\nNext steps:")
        print(f"1. Run diarization: python diarization.py {output_file}")
        print(f"2. Run transcription: python fast-whisper-minimal-segments.py {input_file} {os.path.splitext(output_file)[0]}.segments")
    else:
        print("Processing failed")
