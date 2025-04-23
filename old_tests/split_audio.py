from pydub import AudioSegment
import sys
import os

def split_audio(input_file, duration_minutes=20):
    """
    Split an audio file to keep only the first X minutes
    
    Args:
        input_file: Path to the input audio file
        duration_minutes: Number of minutes to keep from the start
    """
    print(f"Loading audio file: {input_file}")
    audio = AudioSegment.from_file(input_file)
    
    # Calculate duration in milliseconds
    duration_ms = duration_minutes * 60 * 1000
    
    # Get the first X minutes
    print(f"Extracting first {duration_minutes} minutes...")
    shortened_audio = audio[:duration_ms]
    
    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_first_{duration_minutes}min.mp3"
    
    # Export the shortened audio
    print(f"Saving to: {output_file}")
    shortened_audio.export(output_file, format="mp3", bitrate="192k")
    
    print(f"Done! Original duration: {audio.duration_seconds/60:.2f} minutes, New duration: {shortened_audio.duration_seconds/60:.2f} minutes")
    return output_file

if __name__ == "__main__":
    # Get input file from command line or use default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "preprocessed_cardo_1f.mp3"
    
    # Get duration in minutes (default: 20)
    duration_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    output_file = split_audio(input_file, duration_minutes)
    
    print("\nNext steps:")
    print(f"1. Run diarization: python diarization.py {output_file}")
    print(f"2. Run transcription: python fast-whisper-minimal-segments.py {output_file}")
