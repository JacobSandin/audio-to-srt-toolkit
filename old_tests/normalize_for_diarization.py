import sys
import os
from pydub import AudioSegment, effects

def normalize_audio(input_file, output_file):
    """
    Normalize audio levels to ensure both speakers have similar volume
    This helps diarization better distinguish between speakers
    """
    print(f"Loading audio file: {input_file}")
    audio = AudioSegment.from_file(input_file)
    
    print("Original audio level (dBFS):", audio.dBFS)
    
    # Apply normalization (makes quieter parts louder)
    print("Normalizing audio levels...")
    normalized = effects.normalize(audio)
    
    # Apply compression to further balance speaker volumes
    print("Applying compression to balance speaker volumes...")
    compressed = effects.compress_dynamic_range(
        normalized,
        threshold=-20.0,  # Start compressing at -20 dB
        ratio=4.0,        # Stronger compression ratio
        attack=5.0,       # Fast attack to catch speech onsets
        release=50.0      # Slower release for natural sound
    )
    
    print("Processed audio level (dBFS):", compressed.dBFS)
    
    # Export the processed file
    print(f"Saving to: {output_file}")
    compressed.export(output_file, format="mp3", bitrate="192k")
    print("Done!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python normalize_for_diarization.py input.mp3 output.mp3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        sys.exit(1)
    
    success = normalize_audio(input_file, output_file)
    
    if success:
        print("\nNext steps:")
        print(f"1. Run diarization: python diarization.py {output_file}")
        print(f"2. Run transcription: python fast-whisper-minimal-segments.py {input_file} {os.path.splitext(output_file)[0]}.segments")
    else:
        print("Processing failed")
