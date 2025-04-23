from faster_whisper import WhisperModel
import sys
import os

# Get CLI arguments
input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

model_id = "KBLab/kb-whisper-large"
model = WhisperModel(
    model_id,
    device="cuda",
    compute_type="float16",
    download_root="cache"
)

#segments, info = model.transcribe(input_file, condition_on_previous_text=False)
#segments, info = model.transcribe(
#    input_file,
#    language="sv",
#    condition_on_previous_text=False,
#    vad_filter=True
#)
segments, info = model.transcribe(
    input_file,
    language="sv",
    condition_on_previous_text=False,
    vad_filter=False,
    beam_size=5
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

with open(output_file, "w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()

        # Print to console
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}")

        # Write to .srt file
        srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

