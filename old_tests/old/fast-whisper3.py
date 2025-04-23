from faster_whisper import WhisperModel
import sys
import os

input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

# --- Settings ---
LANGUAGE = "sv"
MAX_CHARS = 40
ENGINE_NOISE_TAG = "[Motorljud]"

# --- Model ---
model_id = "KBLab/kb-whisper-large"
model = WhisperModel(
    model_id,
    device="cuda",
    compute_type="float16",
    download_root="cache"
)

segments, info = model.transcribe(
    input_file,
    language=LANGUAGE,
    condition_on_previous_text=False,
    vad_filter=True,
    beam_size=5
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def split_text_by_chars(text, max_chars=40):
    words = text.strip().split()
    lines = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

counter = 1

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in segments:
        text = segment.text.strip()

        # If the segment is empty or too short, assume background noise
        if not text or len(text) < 5:
            text = ENGINE_NOISE_TAG

        split_lines = split_text_by_chars(text, MAX_CHARS)
        duration = (segment.end - segment.start) / max(1, len(split_lines))

        for i, line in enumerate(split_lines):
            start_time = format_timestamp(segment.start + i * duration)
            end_time = format_timestamp(segment.start + (i + 1) * duration)
            print(f"[{segment.start + i * duration:.2f}s -> {segment.start + (i + 1) * duration:.2f}s] {line}")
            srt_file.write(f"{counter}\n{start_time} --> {end_time}\n{line}\n\n")
            counter += 1

