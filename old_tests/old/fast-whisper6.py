from faster_whisper import WhisperModel
import sys
import os

input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

LANGUAGE = "sv"
ENGINE_NOISE_TAG = "[Motorljud]"

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

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

counter = 1
buffer = []
current_start = None

with open(output_file, "w", encoding="utf-8") as srt_file:
    for segment in segments:
        if not segment.words:
            # Insert [Motorljud] for empty segments
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            srt_file.write(f"{counter}\n{start} --> {end}\n{ENGINE_NOISE_TAG}\n\n")
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {ENGINE_NOISE_TAG}")
            counter += 1
            continue

        for word in segment.words:
            if not current_start:
                current_start = word.start
            buffer.append(word)
            
            # End subtitle line if duration is longer than 3s or sentence ends with punctuation
            duration = word.end - current_start
            if duration > 3.0 or word.word.strip()[-1:] in ".!?":
                start = format_timestamp(current_start)
                end = format_timestamp(word.end)
                text = " ".join(w.word for w in buffer).strip()
                srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
                print(f"[{current_start:.2f}s -> {word.end:.2f}s] {text}")
                counter += 1
                buffer = []
                current_start = None

    # Any remaining words at the end
    if buffer:
        start = format_timestamp(current_start)
        end = format_timestamp(buffer[-1].end)
        text = " ".join(w.word for w in buffer).strip()
        srt_file.write(f"{counter}\n{start} --> {end}\n{text}\n\n")
        print(f"[{current_start:.2f}s -> {end:.2f}s] {text}")

