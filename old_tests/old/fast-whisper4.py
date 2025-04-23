from faster_whisper import WhisperModel
import sys
import os

input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

# --- Settings ---
LANGUAGE = "sv"
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

counter = 1

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in segments:
        text = segment.text.strip()

        # Insert [Motorljud] if too short or empty
        if not text or len(text) < 5:
            text = ENGINE_NOISE_TAG

        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)

        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")
        srt_file.write(f"{counter}\n{start_time} --> {end_time}\n{text}\n\n")
        counter += 1

