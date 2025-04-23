from faster_whisper import WhisperModel
import sys
import os

# --- Input/output files ---
input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

# --- Settings ---
LANGUAGE = "sv"
ENGINE_NOISE_TAG = "[Motorljud]"
MAX_GAP = 2.0       # Max silence between words (seconds)
MAX_DURATION = 7.0  # Max subtitle block length (seconds)
MAX_WORDS = 12      # Max words per subtitle line

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
    word_timestamps=True
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# --- Time formatter ---
def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# --- Subtitle builder ---
counter = 1
buffer = []
current_start = None
last_end = None

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in segments:
        for word in segment.words:
            if current_start is None:
                current_start = word.start
                last_end = word.end
                buffer = [word]
                continue

            gap = word.start - last_end
            duration = word.end - current_start

            buffer.append(word)
            last_end = word.end

            should_flush = (
                gap > MAX_GAP or
                duration >= MAX_DURATION or
                len(buffer) >= MAX_WORDS or
                word.word.strip()[-1:] in ".?!"
            )

            if should_flush:
                start = format_timestamp(current_start)
                end = format_timestamp(last_end)
                text = " ".join(w.word.strip() for w in buffer).strip()
                if not text or len(text) < 3:
                    text = ENGINE_NOISE_TAG
                srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
                print(f"[{current_start:.2f}s -> {last_end:.2f}s] {text}")
                counter += 1
                buffer = []
                current_start = None

    # Final flush
    if buffer:
        start = format_timestamp(current_start)
        end = format_timestamp(last_end)
        text = " ".join(w.word.strip() for w in buffer).strip()
        if not text or len(text) < 3:
            text = ENGINE_NOISE_TAG
        srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
        print(f"[{current_start:.2f}s -> {last_end:.2f}s] {text}")

