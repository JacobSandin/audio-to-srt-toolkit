#!/usr/bin/env python3
import warnings
import os
import sys
import logging
import time

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("whisperX-fast-combo")

# --- Suppress all warnings ---
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Redirect stderr to /dev/null to suppress all warnings
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Now import libraries that might generate warnings
import torch
import whisperx
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Disable PyTorch warnings
torch.set_warn_always(False)

# --- Suppress specific warnings and messages ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress specific message patterns
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*speechbrain.*")
warnings.filterwarnings("ignore", message=".*pyannote.*")
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*Special tokens.*")
warnings.filterwarnings("ignore", message=".*weights of the model.*")

# --- Settings ---
LANGUAGE = "sv"
ENGINE_NOISE_TAG = "[Motorljud]"
MAX_GAP = 1.0       # seconds between words before splitting
MAX_DURATION = 5.0  # max subtitle length
MAX_WORDS = 12      # max words per subtitle line
SHORT_SEGMENT_WORDS = 3
GENERATE_INCREMENTAL_SRT = True  # Generate SRT file incrementally during transcription
ALIGN_EACH_CHUNK = False  # Align each chunk immediately after transcription - turning this off for now
CHUNK_SIZE = 15  # Process 15-second chunks at a time

# Default start time for processing (skip initial silence/noise)
DEFAULT_START_TIME = 0  # Start at the beginning of the file

# Model settings
TRANSCRIPTION_MODEL = "KBLab/kb-whisper-large"  # KBLab's Swedish transcription model
ALIGNMENT_MODEL = "KBLab/wav2vec2-large-voxrex-swedish"  # Swedish alignment model

# Chunk size for processing long audio files (in seconds)
# CHUNK_SIZE = 300  # Process 5-minute chunks at a time

# Common transcription errors to fix
CUSTOM_REPLACEMENTS = {
    "tätt": "TET",
    "tet": "TET",
    "tje": "TET",
    "Musik": ENGINE_NOISE_TAG,
    "musik": ENGINE_NOISE_TAG,
    "blöfribensin": "blyfri bensin",
    "blåfri bensin": "blyfri bensin",
    "skjubb": "sju",
    "moglen": "min bil",
    "natta": "något"
}

# Words that likely indicate engine noise or background sounds
NOISE_INDICATORS = ["musik", "Musik", "motor", "ljud", "brus"]

def is_likely_noise(text):
    """Check if text is likely to be noise rather than speech"""
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in NOISE_INDICATORS) or len(text.strip()) < 3

def apply_custom_replacements(text):
    """Apply custom replacements to improve transcription quality"""
    for wrong, right in CUSTOM_REPLACEMENTS.items():
        text = text.replace(wrong, right)
        text = text.replace(wrong.capitalize(), right)
    return text

def ensure_valid_timestamp(timestamp):
    """Ensure timestamp is valid (non-negative)"""
    return max(0.0, timestamp)

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    # Ensure seconds is non-negative
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

# --- Input/output files ---
input_file = sys.argv[1] if len(sys.argv) > 1 else "ToTranscribe.mp3"
output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + ".srt"

# Optional start and end times for processing only a portion of the audio (in seconds)
start_time_sec = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_START_TIME
end_time_sec = float(sys.argv[4]) if len(sys.argv) > 4 else None

# --- Set HuggingFace cache directory to reuse your downloaded model
os.environ["HF_HOME"] = "/home/jacsan/utv/lang/cache"

# --- Load the audio file with torchaudio ---
logger.info("Loading audio file...")
waveform, sample_rate = torchaudio.load(input_file)

# Print audio file information
logger.info(f"Audio file: {input_file}")
logger.info(f"Sample rate: {sample_rate} Hz")
logger.info(f"Channels: {waveform.shape[0]}")
logger.info(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds ({(waveform.shape[1] / sample_rate) / 60:.2f} minutes)")

# Trim audio if start_time_sec or end_time_sec is specified
if start_time_sec > 0 or end_time_sec is not None:
    start_frame = int(start_time_sec * sample_rate)
    end_frame = int(end_time_sec * sample_rate) if end_time_sec is not None else waveform.shape[1]
    logger.info(f"Trimming audio from {start_time_sec:.2f}s to {end_time_sec if end_time_sec is not None else 'end'}")
    waveform = waveform[:, start_frame:end_frame]
    logger.info(f"Trimmed duration: {waveform.shape[1] / sample_rate:.2f} seconds")

# --- Ensure mono channel ---
if waveform.shape[0] > 1:
    logger.info("Stereo audio detected. Converting to mono...")
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono by averaging channels

# --- Resample audio to 16000 Hz if needed ---
required_sample_rate = 16000
if sample_rate != required_sample_rate:
    logger.info(f"Resampling audio from {sample_rate} Hz to {required_sample_rate} Hz...")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=required_sample_rate)
    waveform = resampler(waveform)
    sample_rate = required_sample_rate
    logger.info(f"Resampled audio duration: {waveform.shape[1] / sample_rate:.2f} seconds")

# --- Load Hugging Face model and transcribe ---
logger.info("Loading Hugging Face model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Restore stderr before loading models to see important errors
sys.stderr = original_stderr

try:
    # Load model and processor
    start_time = time.time()
    logger.info(f"Loading {TRANSCRIPTION_MODEL} model for transcription...")
    logger.info(f"Device: {device}, Compute type: {torch_dtype}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        TRANSCRIPTION_MODEL, 
        torch_dtype=torch_dtype, 
        use_safetensors=True, 
        cache_dir="cache"
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(TRANSCRIPTION_MODEL)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    logger.info("Model loaded successfully!")
    logger.info("Transcribing audio...")
    
    # Process in smaller chunks to show progress
    chunk_duration = CHUNK_SIZE  # Process chunks at a time
    total_duration = waveform.shape[1] / sample_rate
    num_chunks = int(total_duration / chunk_duration) + 1
    
    # Initialize SRT counter if generating incrementally
    srt_counter = 1
    if GENERATE_INCREMENTAL_SRT:
        # Open SRT file in write mode to clear any existing content
        with open(output_file, "w", encoding="utf-8") as srt_file:
            pass  # Just clear the file
    
    # Load alignment model if needed
    if ALIGN_EACH_CHUNK:
        logger.info("Loading WhisperX alignment model...")
        # Use CPU for alignment to avoid memory issues
        align_device = "cpu"  # Force CPU for alignment to avoid CUDA out of memory errors
        logger.info(f"Loading alignment model: {ALIGNMENT_MODEL}")
        align_model_start_time = time.time()
        model_a, metadata = whisperx.load_align_model(
            language_code=LANGUAGE,
            device=align_device,
            model_name=ALIGNMENT_MODEL,
            model_dir=os.environ["HF_HOME"]
        )
        logger.info(f"Alignment model loaded in {time.time() - align_model_start_time:.2f} seconds")
    
    # Convert waveform to numpy array for alignment
    audio_np = waveform.squeeze().numpy()
    
    transcription_start_time = time.time()
    
    for i in range(num_chunks):
        chunk_start = i * chunk_duration
        chunk_end = min((i + 1) * chunk_duration, total_duration)
        
        # Skip if we're at the end
        if chunk_start >= total_duration:
            break
        
        logger.info(f"Processing chunk {i+1}/{num_chunks} ({chunk_start:.2f}s to {chunk_end:.2f}s)...")
        
        # Extract chunk
        chunk_start_frame = int(chunk_start * sample_rate)
        chunk_end_frame = int(chunk_end * sample_rate)
        chunk_waveform = waveform[:, chunk_start_frame:chunk_end_frame]
        
        # Save chunk to temporary file
        chunk_file = f"temp_chunk_{i}.wav"
        torchaudio.save(chunk_file, chunk_waveform, sample_rate)
        
        # Transcribe chunk
        logger.info(f"Transcribing chunk {i+1}/{num_chunks}...")
        chunk_result = pipe(
            chunk_file,
            chunk_length_s=10,  # Use smaller chunks for more precise timestamps
            stride_length_s=[1, 1],  # Small overlap for better results
            return_timestamps=True,
            generate_kwargs={"task": "transcribe", "language": LANGUAGE}
        )
        
        # Process results
        chunk_segments = []
        
        if "chunks" in chunk_result:
            for chunk in chunk_result["chunks"]:
                # Skip empty or whitespace-only segments
                if not chunk["text"] or chunk["text"].strip() == "":
                    continue
                    
                # Check if timestamp exists and is valid
                if "timestamp" in chunk and chunk["timestamp"] is not None and len(chunk["timestamp"]) == 2 and chunk["timestamp"][0] is not None and chunk["timestamp"][1] is not None:
                    # Calculate global timestamp
                    global_start = chunk["timestamp"][0] + chunk_start + start_time_sec
                    global_end = chunk["timestamp"][1] + chunk_start + start_time_sec
                    
                    segment = {
                        "start": ensure_valid_timestamp(global_start),
                        "end": ensure_valid_timestamp(global_end),
                        "text": chunk["text"].strip()
                    }
                    
                    # Ensure end is after start
                    if segment["end"] <= segment["start"]:
                        segment["end"] = segment["start"] + 0.1
                        
                    chunk_segments.append(segment)
                    logger.info(f"Transcribed: [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
                else:
                    # Handle missing timestamp by using chunk boundaries
                    segment = {
                        "start": chunk_start + start_time_sec,
                        "end": chunk_end + start_time_sec,
                        "text": chunk["text"].strip()
                    }
                    chunk_segments.append(segment)
                    logger.info(f"Transcribed (no timestamp): [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
        else:
            # Handle result without timestamps
            # Skip if text is empty or just whitespace
            if chunk_result["text"] and chunk_result["text"].strip() != "":
                segment = {
                    "start": chunk_start + start_time_sec,
                    "end": chunk_end + start_time_sec,
                    "text": chunk_result["text"].strip()
                }
                chunk_segments.append(segment)
                logger.info(f"Transcribed: [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text'][:100]}...")
        
        # Skip if no segments were found
        if not chunk_segments:
            logger.info(f"No segments found in chunk {i+1}/{num_chunks}, skipping...")
            continue
        
        # Align this chunk if enabled
        aligned_words = []
        if ALIGN_EACH_CHUNK:
            logger.info(f"Aligning chunk {i+1}/{num_chunks}...")
            # Extract audio for this chunk
            chunk_audio_np = audio_np[chunk_start_frame:chunk_end_frame]
            
            # Create a copy of the segments with local timestamps (relative to chunk start)
            local_segments = []
            for segment in chunk_segments:
                local_segment = segment.copy()
                # Convert global timestamps to local (relative to chunk)
                local_start = segment["start"] - start_time_sec
                local_end = segment["end"] - start_time_sec
                
                # Ensure local timestamps are non-negative
                local_segment["start"] = max(0, local_start)
                local_segment["end"] = max(0, local_end)
                local_segments.append(local_segment)
            
            # Align the transcription for this chunk
            try:
                chunk_aligned = whisperx.align(local_segments, model_a, metadata, chunk_audio_np, align_device)
                aligned_words = chunk_aligned.get("word_segments", [])
                logger.info(f"Aligned {len(aligned_words)} words in chunk {i+1}/{num_chunks}")
                
                # Log some of the aligned words
                if aligned_words:
                    logger.info(f"First few aligned words in chunk {i+1}/{num_chunks}:")
                    for j, word in enumerate(aligned_words[:5]):
                        # Convert local timestamps back to global
                        global_start = word["start"] + start_time_sec
                        global_end = word["end"] + start_time_sec
                        
                        word["start"] = ensure_valid_timestamp(global_start)
                        word["end"] = ensure_valid_timestamp(global_end)
                        
                        # Ensure end is after start
                        if word["end"] <= word["start"]:
                            word["end"] = word["start"] + 0.1  # Add a small duration
                        
                        logger.info(f"  Word {j+1}: '{word['word']}' at {word['start']:.2f}s -> {word['end']:.2f}s")
            except Exception as e:
                logger.error(f"Error aligning chunk {i+1}/{num_chunks}: {e}")
                # Fall back to using the transcription segments
                logger.info(f"Falling back to transcription timestamps for chunk {i+1}/{num_chunks}")
                aligned_words = []
        
        # Generate SRT entries for this chunk
        if GENERATE_INCREMENTAL_SRT:
            with open(output_file, "a", encoding="utf-8") as srt_file:
                if ALIGN_EACH_CHUNK and aligned_words:
                    # Process aligned words into subtitles
                    buffer = []
                    start_time = None
                    last_end = None
                    
                    for word in aligned_words:
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
                            
                            # Check if this is likely noise
                            if is_likely_noise(text):
                                text = ENGINE_NOISE_TAG
                            
                            # Format timestamps
                            start_str = format_timestamp(start_time)
                            end_str = format_timestamp(last_end)
                            
                            # Write SRT entry
                            srt_file.write(f"{srt_counter}\n{start_str} --> {end_str}\n{text}\n\n")
                            logger.info(f"SRT Entry: [{start_time:.2f}s -> {last_end:.2f}s] {text}")
                            srt_counter += 1
                            
                            buffer = []
                            start_time = None
                    
                    # Final flush for this chunk
                    if buffer:
                        text = " ".join(w["word"].strip() for w in buffer).strip()
                        text = apply_custom_replacements(text)
                        
                        # Check if this is likely noise
                        if is_likely_noise(text):
                            text = ENGINE_NOISE_TAG
                        
                        # Format timestamps
                        start_str = format_timestamp(start_time)
                        end_str = format_timestamp(last_end)
                        
                        # Write SRT entry
                        srt_file.write(f"{srt_counter}\n{start_str} --> {end_str}\n{text}\n\n")
                        logger.info(f"SRT Entry: [{start_time:.2f}s -> {last_end:.2f}s] {text}")
                        srt_counter += 1
                else:
                    # Use transcription segments directly
                    for segment in chunk_segments:
                        # Format timestamps
                        start_str = format_timestamp(segment["start"])
                        end_str = format_timestamp(segment["end"])
                        
                        # Apply custom replacements
                        text = apply_custom_replacements(segment["text"])
                        
                        # Check if this is likely noise
                        if is_likely_noise(text):
                            text = ENGINE_NOISE_TAG
                        
                        # Write SRT entry
                        srt_file.write(f"{srt_counter}\n{start_str} --> {end_str}\n{text}\n\n")
                        logger.info(f"SRT Entry: [{segment['start']:.2f}s -> {segment['end']:.2f}s] {text}")
                        srt_counter += 1
            
            logger.info(f"Added entries to {output_file} for chunk {i+1}/{num_chunks}")
    
    transcription_time = time.time() - transcription_start_time
    logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
    logger.info(f"Subtitles generated: {output_file}")

except Exception as e:
    logger.error(f"Error loading or using model: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

# Reset HF_HOME
del os.environ["HF_HOME"]

# Clean up temporary files
try:
    for i in range(num_chunks):
        chunk_file = f"temp_chunk_{i}.wav"
        os.remove(chunk_file)
except:
    pass
