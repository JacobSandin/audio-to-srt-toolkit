from faster_whisper import WhisperModel
import sys
import os
import torchaudio
import torch

# --- Input/output files ---
input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

# Process specific time range if specified
start_time_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 0
end_time_sec = float(sys.argv[4]) if len(sys.argv) > 4 else None

# --- Settings ---
LANGUAGE = "sv"
ENGINE_NOISE_TAG = "[Motorljud]"
MAX_GAP = 1.0       # seconds between words before splitting
MAX_DURATION = 5.0  # max subtitle length
MAX_SEGMENT_DURATION = 10.0  # maximum duration for a single segment in seconds
MAX_WORDS = 12      # max words per subtitle line
SHORT_SEGMENT_WORDS = 3
SHORT_DURATION_FALLBACK = 1.5  # seconds
MODEL_NAME = "KBLab/kb-whisper-large"  # Model to use for transcription
#MODEL_NAME = "large"  # OpenAI's Whisper model
CHUNK_SIZE = 300    # Process audio in 5-minute chunks
CHUNK_OVERLAP = 60  # Overlap between chunks in seconds
TIMESTAMP_OFFSET = 0.07  # Add a small offset to start times to fix alignment
USE_VAD = True     # Disable VAD filtering to capture all audio
VAD_THRESHOLD = 0.0  # Lower threshold for VAD (only used if USE_VAD is True)

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

# --- Load audio ---
print(f"Loading audio file: {input_file}")
waveform, sample_rate = torchaudio.load(input_file)

# Get audio duration
audio_duration = waveform.shape[1] / sample_rate
print(f"Audio duration: {audio_duration:.2f} seconds")

# Apply time range if specified
if start_time_sec > 0 or end_time_sec is not None:
    # Calculate frames
    start_frame = int(start_time_sec * sample_rate)
    end_frame = int(end_time_sec * sample_rate) if end_time_sec is not None else waveform.shape[1]
    
    print(f"Trimming audio from {start_time_sec:.2f}s to {end_time_sec if end_time_sec is not None else 'end'}")
    waveform = waveform[:, start_frame:end_frame]
    audio_duration = waveform.shape[1] / sample_rate
    print(f"Trimmed audio duration: {audio_duration:.2f} seconds")

# --- Load model ---
print(f"Loading model: {MODEL_NAME}")
model = WhisperModel(
    MODEL_NAME,
    device="cuda",
    compute_type="float16",
    download_root="cache"
)

# --- Process in chunks ---
all_segments = []
num_chunks = int(audio_duration / (CHUNK_SIZE - CHUNK_OVERLAP)) + 1

for i in range(num_chunks):
    chunk_start = i * (CHUNK_SIZE - CHUNK_OVERLAP)
    chunk_end = min(chunk_start + CHUNK_SIZE, audio_duration)
    
    # Skip if we're at the end
    if chunk_start >= audio_duration:
        break
    
    print(f"\nProcessing chunk {i+1}/{num_chunks} ({chunk_start:.2f}s to {chunk_end:.2f}s)...")
    
    # Extract chunk audio
    chunk_start_frame = int(chunk_start * sample_rate)
    chunk_end_frame = int(chunk_end * sample_rate)
    chunk_waveform = waveform[:, chunk_start_frame:chunk_end_frame]
    
    # Save to temporary file
    temp_audio_file = f"temp_chunk_{i}.wav"
    torchaudio.save(temp_audio_file, chunk_waveform, sample_rate)
    
    # --- Transcribe this chunk ---
    segments, info = model.transcribe(
        temp_audio_file,
        language=LANGUAGE,
        condition_on_previous_text=False,
        vad_filter=USE_VAD,
        vad_parameters={"threshold": VAD_THRESHOLD} if USE_VAD else None,
        beam_size=5,
        word_timestamps=True
    )
    
    # Log language detection for the first chunk
    if i == 0:
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    
    # Convert segments to list for easier processing
    segment_list = list(segments)
    print(f"Found {len(segment_list)} segments in chunk {i+1}/{num_chunks}")
    
    # Adjust timestamps to global time
    for segment in segment_list:
        segment.start += chunk_start
        segment.end += chunk_start
    
    # Add to all segments, excluding overlap with previous chunk
    overlap_threshold = chunk_start + CHUNK_OVERLAP if i > 0 else 0
    for segment in segment_list:
        if segment.start >= overlap_threshold:
            all_segments.append(segment)
    
    # Clean up temporary file
    try:
        os.remove(temp_audio_file)
    except:
        pass

print(f"\nProcessing {len(all_segments)} total segments")

# --- Process segments ---
processed_segments = []

# Process each segment
for segment in all_segments:
    text = segment.text.strip()
    text = apply_custom_replacements(text)
    
    # Skip empty segments
    if not text or len(text) < 3:
        continue
    
    # Skip hallucinated text
    if is_hallucinated_text(text):
        print(f"Skipping hallucinated text: '{text}'")
        continue
    
    start_sec = segment.start
    end_sec = segment.end
    duration = end_sec - start_sec
    word_count = len(text.split())
    
    # Skip segments that are too long
    if duration > MAX_SEGMENT_DURATION:
        print(f"Skipping segment that's too long: {duration:.2f}s > {MAX_SEGMENT_DURATION:.2f}s - '{text}'")
        continue
    
    # Adjust timestamps for short segments
    if word_count <= SHORT_SEGMENT_WORDS:
        estimated_duration = min(1.5 + word_count * 0.3, duration)
        corrected_start = end_sec - estimated_duration
        if corrected_start > start_sec:
            print(f"Adjusting start from {start_sec:.2f} u2192 {corrected_start:.2f} for short segment ({word_count} words)")
            start_sec = corrected_start
    
    # Add to processed segments
    processed_segments.append({
        "text": text,
        "start": start_sec,
        "end": end_sec
    })

# --- Merge segments that are too close ---
merged_segments = []
if processed_segments:
    current = processed_segments[0]
    
    for next_segment in processed_segments[1:]:
        gap = next_segment["start"] - current["end"]
        merged_duration = next_segment["end"] - current["start"]
        merged_word_count = len(current["text"].split()) + len(next_segment["text"].split())
        
        # Merge if the gap is small and the result isn't too long
        if gap < MAX_GAP and merged_duration < MAX_DURATION and merged_word_count <= MAX_WORDS:
            # Merge the segments
            current["text"] = f"{current['text']} {next_segment['text']}"
            current["end"] = next_segment["end"]
            print(f"Merged segments with gap {gap:.2f}s: {current['text']}")
        else:
            # Add current to merged list and move to next
            merged_segments.append(current)
            current = next_segment
    
    # Add the last segment
    merged_segments.append(current)

# --- Write SRT file ---
print(f"Writing {len(merged_segments)} segments to {output_file}")
counter = 1

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in merged_segments:
        text = segment["text"]
        
        # Apply final text processing
        if not text or len(text) < 3:
            text = ENGINE_NOISE_TAG
        
        # Get timestamps
        start_sec = segment["start"] + TIMESTAMP_OFFSET  # Add offset to start time
        end_sec = segment["end"]
        
        # Add offset if we're processing a specific time range
        if start_time_sec > 0:
            start_sec += start_time_sec
            end_sec += start_time_sec
        
        # Format timestamps
        start = format_timestamp(start_sec)
        end = format_timestamp(end_sec)
        
        # Write to SRT file
        srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
        print(f"[{start_sec:.2f}s -> {end_sec:.2f}s] {text}")
        counter += 1

print(f"Successfully generated SRT file: {output_file}")
