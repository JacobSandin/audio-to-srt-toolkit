from faster_whisper import WhisperModel
import sys
import os
import torchaudio
import torch
from pydub import AudioSegment
import tempfile

# --- Input/output files ---
input_file = sys.argv[1] if len(sys.argv) > 1 else "preprocessed_cardo_1f.mp3"
# Check if a segments file was provided, otherwise look for a .segments file with the same name as the input file
if len(sys.argv) > 2:
    segments_file = sys.argv[2]
else:
    default_segments = os.path.splitext(input_file)[0] + ".segments"
    if os.path.exists(default_segments):
        segments_file = default_segments
    else:
        segments_file = "segments.txt"  # Fallback to the old name

output_file = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(input_file)[0] + "_with_speakers.srt"

# --- Settings ---
LANGUAGE = "sv"
ENGINE_NOISE_TAG = "[Motorljud]"
MAX_GAP = 1.0       # seconds between words before splitting
MAX_DURATION = 5.0  # max subtitle length
MAX_WORDS = 10      # max words per subtitle line
SHORT_SEGMENT_WORDS = 3
SHORT_DURATION_FALLBACK = 1.5  # seconds
MODEL_NAME = "KBLab/kb-whisper-large"  # Model to use for transcription
TIMESTAMP_OFFSET = 0.07  # Add a small offset to start times to fix alignment

# --- Speaker mapping (customize as needed) ---
SPEAKER_NAMES = {
    "SPEAKER_00": "Person 1",
    "SPEAKER_01": "Person 2"
}

# --- Filter for hallucinated text ---
HALLUCINATION_PATTERNS = [
    "textning.nu",
    "undertexter från",
    "amara.org",
    "tack till elever",
    "värmlands universi",
    "gemenskapen",
    "Stina Hedin",
    "btistudio",
    "&lt;i&gt;"
]

def is_hallucinated_text(text: str) -> bool:
    """Check if text matches common hallucination patterns."""
    text_lower = text.lower()
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in text_lower:
            return True
    return False

# --- Custom replacements ---
CUSTOM_REPLACEMENTS = {
    "test": "TET",
    "herr Bösse": "Mr. Bo"
}

def apply_custom_replacements(text: str) -> str:
    for wrong, right in CUSTOM_REPLACEMENTS.items():
        text = text.replace(wrong, right)
        text = text.replace(wrong.capitalize(), right)
    return text

# --- Time formatter ---
def format_timestamp(seconds: float) -> str:
    # Ensure seconds is non-negative
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# --- Parse segments file ---
print(f"Reading speaker segments from: {segments_file}")
segments = []
with open(segments_file, "r") as f:
    for line in f:
        if "Speaker" in line and "from" in line and "to" in line:
            parts = line.strip().split()
            speaker = parts[1]
            # Handle time values with or without 's' suffix
            start_str = parts[3]
            if start_str.endswith('s'):
                start_time = float(start_str[:-1])  # Remove trailing 's'
            else:
                start_time = float(start_str)
                
            end_str = parts[5]
            if end_str.endswith('s'):
                end_time = float(end_str[:-1])  # Remove trailing 's'
            else:
                end_time = float(end_str)
                
            segments.append({
                "speaker": speaker,
                "start": start_time,
                "end": end_time
            })

print(f"Found {len(segments)} speaker segments")

# --- Load audio ---
print(f"Loading audio file: {input_file}")
audio = AudioSegment.from_file(input_file)
audio_duration = audio.duration_seconds
print(f"Audio duration: {audio_duration:.2f} seconds")

# --- Load model ---
print(f"Loading model: {MODEL_NAME}")
model = WhisperModel(
    MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16",
    download_root="cache"
)

# --- Process each speaker segment ---
processed_segments = []

for i, seg in enumerate(segments):
    print(f"Processing segment {i+1}/{len(segments)} - Speaker {seg['speaker']} from {seg['start']:.2f}s to {seg['end']:.2f}s")
    
    # Extract audio segment
    start_ms = int(seg["start"] * 1000)
    end_ms = int(seg["end"] * 1000)
    segment_audio = audio[start_ms:end_ms]
    
    # Skip segments that are too short (less than 0.5 seconds)
    if seg["end"] - seg["start"] < 0.1:
        print(f"Skipping segment that's too short: {seg['end'] - seg['start']:.2f}s")
        continue
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        segment_audio.export(temp.name, format="wav")
        temp_filename = temp.name
    
    # Transcribe
    transcription, _ = model.transcribe(
        temp_filename, 
        language=LANGUAGE,
        beam_size=5,
        word_timestamps=True
    )
    
    # Get the text
    transcription_list = list(transcription)
    
    # Clean up
    os.unlink(temp_filename)
    
    # Skip if no transcription
    if not transcription_list:
        print(f"No transcription for segment {i+1}")
        continue
    
    # Process each transcription segment
    for trans_seg in transcription_list:
        text = trans_seg.text.strip()
        text = apply_custom_replacements(text)
        
        # Skip empty segments
        if not text or len(text) < 3:
            continue
        
        # Skip hallucinated text
        if is_hallucinated_text(text):
            print(f"Skipping hallucinated text: '{text}'")
            continue
        
        # Calculate global timestamps
        start_sec = seg["start"] + trans_seg.start
        end_sec = seg["start"] + trans_seg.end
        
        # Add to processed segments
        processed_segments.append({
            "text": text,
            "start": start_sec,
            "end": end_sec,
            "speaker": seg["speaker"]
        })

# --- Sort segments by start time ---
processed_segments.sort(key=lambda x: x["start"])

# --- Write SRT file ---
print(f"Writing {len(processed_segments)} segments to {output_file}")
counter = 1

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in processed_segments:
        text = segment["text"]
        speaker = SPEAKER_NAMES.get(segment["speaker"], segment["speaker"])
        
        # Apply final text processing
        if not text or len(text) < 3:
            text = ENGINE_NOISE_TAG
        
        # Get timestamps
        start_sec = segment["start"] + TIMESTAMP_OFFSET  # Add offset to start time
        end_sec = segment["end"]
        
        # Format timestamps
        start = format_timestamp(start_sec)
        end = format_timestamp(end_sec)
        
        # Write to SRT file
        srt_file.write(f"{counter}\n{start} --> {end}\n{speaker}: {text}\n\n")
        print(f"[{start_sec:.2f}s -> {end_sec:.2f}s] {speaker}: {text}")
        counter += 1

print(f"Successfully generated SRT file: {output_file}")
