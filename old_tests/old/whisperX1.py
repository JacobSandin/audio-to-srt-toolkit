import os
import sys
import torch
import whisperx

# --- Settings ---
os.environ["HF_HOME"] = "/home/jacsan/utv/lang/wcache"  # reuse KBLab model cache

LANGUAGE = "sv"
ENGINE_NOISE_TAG = "[Motorljud]"
MAX_GAP = 1.0       # seconds between words before splitting
MAX_DURATION = 5.0  # max subtitle length
MAX_WORDS = 12      # max words per subtitle line
SHORT_SEGMENT_WORDS = 3

CUSTOM_REPLACEMENTS = {
    "tätt": "TET",
    "tet": "TET",
    "tje": "TET"
}

def apply_custom_replacements(text):
    for wrong, right in CUSTOM_REPLACEMENTS.items():
        text = text.replace(wrong, right)
        text = text.replace(wrong.capitalize(), right)
    return text

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# --- Input/output files ---
input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

# --- WhisperX: Load and transcribe
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16"

print("🔄 Loading KBLab Whisper model...")
model = whisperx.load_model("KBLab/kb-whisper-large", device, language=LANGUAGE, compute_type=compute_type)

print("🔊 Transcribing audio...")
audio = whisperx.load_audio(input_file)
result = model.transcribe(audio)

print("🔁 Loading align model...")
model_a, metadata = whisperx.load_align_model(
    language_code=LANGUAGE,
    device=device,
    model_name="KBLab/wav2vec2-large-voxrex-swedish",
    model_dir=os.environ["HF_HOME"]
)

print("📐 Aligning words...")
aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

# --- Build SRT
counter = 1
buffer = []
start_time = None
last_end = None

with open(output_file, "w", encoding="utf-8") as srt_file:
    for word in aligned["word_segments"]:
        if start_time is None:
            start_time = word["start"]
            last_end = word["end"]
            buffer = [word]
            continue

        gap = word["start"] - last_end
        duration = word["end"] - start_time

        buffer.append(word)
        last_end = word["end"]

        should_flush = (
            gap > MAX_GAP or
            duration >= MAX_DURATION or
            len(buffer) >= MAX_WORDS or
            word["word"].strip()[-1:] in ".?!"
        )

        if should_flush:
            text = " ".join(w["word"].strip() for w in buffer).strip()
            text = apply_custom_replacements(text)
            if not text or len(text) < 3:
                text = ENGINE_NOISE_TAG

            word_count = len(buffer)
            raw_duration = buffer[-1]["end"] - buffer[0]["start"]

            if raw_duration > 10.0 and word_count <= SHORT_SEGMENT_WORDS:
                est_duration = min(1.5 + word_count * 0.3, raw_duration)
                corrected_start = buffer[-1]["end"] - est_duration
                if corrected_start > buffer[0]["start"]:
                    start_time = corrected_start
                else:
                    start_time = buffer[0]["start"]
            else:
                start_time = buffer[0]["start"]

            start = format_timestamp(start_time)
            end = format_timestamp(buffer[-1]["end"])
            srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
            print(f"[{start_time:.2f}s -> {buffer[-1]['end']:.2f}s] {text}")
            counter += 1
            buffer = []
            start_time = None

    # Final flush
    if buffer:
        text = " ".join(w["word"].strip() for w in buffer).strip()
        text = apply_custom_replacements(text)
        if not text or len(text) < 3:
            text = ENGINE_NOISE_TAG

        word_count = len(buffer)
        raw_duration = buffer[-1]["end"] - buffer[0]["start"]

        if raw_duration > 10.0 and word_count <= SHORT_SEGMENT_WORDS:
            est_duration = min(1.5 + word_count * 0.3, raw_duration)
            corrected_start = buffer[-1]["end"] - est_duration
            if corrected_start > buffer[0]["start"]:
                start_time = corrected_start
            else:
                start_time = buffer[0]["start"]
        else:
            start_time = buffer[0]["start"]

        start = format_timestamp(start_time)
        end = format_timestamp(buffer[-1]["end"])
        srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
        print(f"[{start_time:.2f}s -> {buffer[-1]['end']:.2f}s] {text}")

# Reset HF_HOME
del os.environ["HF_HOME"]

