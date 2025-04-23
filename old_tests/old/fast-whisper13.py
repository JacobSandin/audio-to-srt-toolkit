from faster_whisper import WhisperModel
import sys
import os
import numpy as np
import torch
import torchaudio
import logging

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("fast-whisper")

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
MAX_WORDS = 12      # max words per subtitle line
SHORT_SEGMENT_WORDS = 3
SHORT_DURATION_FALLBACK = 1.5  # seconds
MIN_SEGMENT_DURATION = 0.5  # minimum segment duration in seconds
VAD_FILTER = True    # Use Voice Activity Detection to filter out non-speech
#MODEL_NAME = "large"  # OpenAI Whisper "large"
MODEL_NAME = "KBLab/kb-whisper-large"  # Model to use for transcription

# --- Custom replacements ---
CUSTOM_REPLACEMENTS = {
    "tätt": "TET",
    "tet": "TET",
    "tje": "TET"
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

# --- Load audio and trim if needed ---
if start_time_sec > 0 or end_time_sec is not None:
    logger.info(f"Loading audio file for trimming: {input_file}")
    waveform, sample_rate = torchaudio.load(input_file)
    
    # Calculate frames
    start_frame = int(start_time_sec * sample_rate)
    end_frame = int(end_time_sec * sample_rate) if end_time_sec is not None else waveform.shape[1]
    
    logger.info(f"Trimming audio from {start_time_sec:.2f}s to {end_time_sec if end_time_sec is not None else 'end'}")
    waveform = waveform[:, start_frame:end_frame]
    
    # Save to temporary file
    temp_audio_file = "temp_trimmed_audio.wav"
    torchaudio.save(temp_audio_file, waveform, sample_rate)
    
    # Use the temporary file for transcription
    transcription_file = temp_audio_file
    logger.info(f"Saved trimmed audio to {temp_audio_file}")
else:
    transcription_file = input_file

# --- Load model ---
logger.info(f"Loading model: {MODEL_NAME}")
model = WhisperModel(
    MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "float32",
    download_root="cache"
)

# --- Transcribe ---
logger.info(f"Transcribing audio: {transcription_file}")
segments, info = model.transcribe(
    transcription_file,
    language=LANGUAGE,
    condition_on_previous_text=False,
    vad_filter=VAD_FILTER,
    beam_size=5,
    word_timestamps=True  # Get word-level timestamps
)

logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

# --- Process segments ---
processed_segments = []

# Convert segments to list for easier processing
segment_list = list(segments)
logger.info(f"Processing {len(segment_list)} segments")

# Process each segment
for segment in segment_list:
    text = segment.text.strip()
    text = apply_custom_replacements(text)
    
    # Skip empty segments
    if not text or len(text) < 3:
        continue
    
    start_sec = segment.start
    end_sec = segment.end
    duration = end_sec - start_sec
    word_count = len(text.split())
    
    # Skip segments that are too short
    if duration < MIN_SEGMENT_DURATION:
        logger.info(f"Skipping segment that's too short: {duration:.2f}s < {MIN_SEGMENT_DURATION:.2f}s")
        continue
    
    # Adjust timestamps for short segments
    if word_count <= SHORT_SEGMENT_WORDS:
        estimated_duration = min(1.5 + word_count * 0.3, duration)
        corrected_start = end_sec - estimated_duration
        if corrected_start > start_sec:
            logger.info(f"Adjusting start from {start_sec:.2f} → {corrected_start:.2f} for short segment ({word_count} words)")
            start_sec = corrected_start
    
    # Add to processed segments
    processed_segments.append({
        "text": text,
        "start": start_sec,
        "end": end_sec,
        "words": segment.words
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
            current["words"] = current["words"] + next_segment["words"]
            logger.info(f"Merged segments with gap {gap:.2f}s: {current['text']}")
        else:
            # Add current to merged list and move to next
            merged_segments.append(current)
            current = next_segment
    
    # Add the last segment
    merged_segments.append(current)

# --- Write SRT file ---
logger.info(f"Writing {len(merged_segments)} segments to {output_file}")
counter = 1

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in merged_segments:
        text = segment["text"]
        
        # Apply final text processing
        if not text or len(text) < 3:
            text = ENGINE_NOISE_TAG
        
        # Get timestamps
        start_sec = segment["start"]
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
        logger.info(f"[{start_sec:.2f}s -> {end_sec:.2f}s] {text}")
        counter += 1

logger.info(f"Successfully generated SRT file: {output_file}")

# Clean up temporary file if created
if start_time_sec > 0 or end_time_sec is not None:
    try:
        os.remove(temp_audio_file)
        logger.info(f"Removed temporary file: {temp_audio_file}")
    except:
        pass
