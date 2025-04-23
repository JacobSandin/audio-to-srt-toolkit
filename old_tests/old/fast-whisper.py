#### faster-whisper model ####
from faster_whisper import WhisperModel

model_id = "KBLab/kb-whisper-large"
model = WhisperModel(
    model_id,
    device="cuda",
    compute_type="float16",
    download_root="cache", # cache directory
    # condition_on_previous_text = False # Can reduce hallucinations if we don't use prompts
)

# Transcribe audio.wav (convert to 16khz mono wav first via ffmpeg)
segments, info = model.transcribe("./ToTranscribe.mp3", condition_on_previous_text=False)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

