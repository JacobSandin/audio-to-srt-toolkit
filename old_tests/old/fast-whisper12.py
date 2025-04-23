from faster_whisper import WhisperModel
import sys
import os

# --- Input/output files ---
input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

# --- Settings ---
LANGUAGE = "sv"
ENGINE_NOISE_TAG = "[Motorljud]"
MAX_SEGMENT_DURATION = 10.0  # seconds
SHORT_SEGMENT_WORDS = 3
SHORT_DURATION_FALLBACK = 1.5  # seconds

# --- TET & MyWorld replacements ---
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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# --- Load model ---
model = WhisperModel(
    "KBLab/kb-whisper-large",
    device="cuda",
    compute_type="float16",
    download_root="cache"
)

segments, info = model.transcribe(
    input_file,
    language=LANGUAGE,
    condition_on_previous_text=False,
    vad_filter=True,
    beam_size=5,
    word_timestamps=True  # still used for segment alignment
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# --- SRT output ---
counter = 1

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in segments:
        text = segment.text.strip()
        text = apply_custom_replacements(text)
        if not text or len(text) < 3:
            text = ENGINE_NOISE_TAG

        start_sec = segment.start
        end_sec = segment.end
        duration = end_sec - start_sec
        word_count = len(text.split())

        if word_count <= SHORT_SEGMENT_WORDS:
            estimated_duration = min(1.5 + word_count * 0.3, duration)
            corrected_start = end_sec - estimated_duration
            if corrected_start > start_sec:
                print(f"⚠️  Adjusting start from {start_sec:.2f} → {corrected_start:.2f} for short segment ({word_count} words)")
                start_sec = corrected_start
        
        start = format_timestamp(start_sec)
        end = format_timestamp(end_sec)

        srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
        print(f"[{start_sec:.2f}s -> {end_sec:.2f}s] {text}")
        counter += 1

