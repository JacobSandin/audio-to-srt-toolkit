import whisperx
import sys

device = "cuda"
audio_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model(
    "KBLab/kb-whisper-large", device, compute_type=compute_type, download_root="cache"  # cache_dir
)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])  # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device,
    model_name="KBLab/wav2vec2-large-voxrex-swedish",
    model_dir="cache",  # cache_dir
)
result = whisperx.align(
    result["segments"], model_a, metadata, audio, device, return_char_alignments=False
)

print(result["segments"])  # word level timestamps after alignment

