from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import sys
import os
import time


# Get input filename from command line argument or use default
input_file = sys.argv[1] if len(sys.argv) > 1 else "preprocessed_cardo_1b.mp3"

# Generate output filename based on input filename
output_file = os.path.splitext(input_file)[0] + ".segments"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Processing file: {input_file}")
print(f"Output will be saved to: {output_file}")

# Load the latest diarization pipeline (3.1 version)
pipeline = Pipeline.from_pretrained(
    "tensorlake/speaker-diarization-3.1",
    use_auth_token="[TOKEN]"
)

# Optimize GPU usage
if torch.cuda.is_available():
    print("Optimizing GPU usage...")
    # Move pipeline to GPU
    pipeline.to(torch.device("cuda"))
    
    # Enable TF32 for faster processing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set benchmark mode for faster processing with fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    # Increase batch size for better GPU utilization
    try:
        if hasattr(pipeline, "batch_size"):
            pipeline.batch_size = 32  # Larger batch size for better GPU utilization
            print(f"Set batch_size to {pipeline.batch_size}")
    except Exception as e:
        print(f"Could not set batch size: {e}")
    
    # Try to allocate more GPU memory
    try:
        # Reserve more GPU memory upfront
        torch.cuda.empty_cache()
        # Pre-allocate memory to avoid fragmentation
        torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
        print("Reserved 95% of GPU memory for processing")
    except Exception as e:
        print(f"Could not optimize GPU memory: {e}")
        
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("GPU not available, using CPU only")

# Additional pipeline properties for Swedish dialect separation
try:
    # Set min_speakers and max_speakers to help with Swedish dialects
    print("Setting additional pipeline properties...")
    pipeline.min_speakers = 2  # At least 2 speakers (the two Swedish dialects)
    pipeline.max_speakers = 3  # At most 3 speakers (including background)
    
    # Try to set segmentation threshold for better speaker turns detection
    if hasattr(pipeline, "segmentation_threshold"):
        pipeline.segmentation_threshold = 0.5  # More sensitive to speaker changes
        print("Set segmentation_threshold to 0.5")
        
    # Try to set clustering threshold for better speaker separation
    if hasattr(pipeline, "clustering_threshold"):
        pipeline.clustering_threshold = 0.65  # Much lower = force more speakers
        print("Set clustering_threshold to 0.65")
    
    # Set more aggressive speaker embedding parameters if available
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = 32  # Process more embeddings at once
        print("Set embedding_batch_size to 32")
    
    # Set more aggressive segmentation parameters if available
    if hasattr(pipeline, "segmentation"):
        if isinstance(pipeline.segmentation, dict):
            pipeline.segmentation["min_duration_off"] = 0.1  # Shorter silence
            print("Set min_duration_off to 0.1")
    
    print("Pipeline properties set successfully")
except Exception as e:
    print(f"Could not set some pipeline properties: {e}")
    print("Continuing with default properties")

# Start timing
start_time = time.time()
print("Starting diarization...")

# Preprocess the audio to enhance speaker differences for Swedish dialects
print("Preprocessing audio to enhance speaker differences...")
try:
    from pydub import AudioSegment
    import numpy as np
    from scipy.signal import butter, filtfilt
    
    # Load the audio
    print("Loading audio file...")
    audio = AudioSegment.from_file(input_file)
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Get audio array
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    
    # Apply a high-pass filter to emphasize higher frequencies (where dialects differ)
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    def apply_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    # Apply high-pass filter (1000Hz) to emphasize consonant differences in dialects
    print("Applying high-pass filter to emphasize dialect differences...")
    filtered_samples = apply_highpass_filter(samples, cutoff=1000, fs=sample_rate)
    
    # Normalize and enhance the filtered audio
    filtered_samples = filtered_samples.astype(np.float32)
    filtered_samples = filtered_samples / np.max(np.abs(filtered_samples))
    
    # Enhance the differences (mild compression)
    filtered_samples = np.sign(filtered_samples) * (np.abs(filtered_samples) ** 0.8)
    
    # Convert back to integer samples
    filtered_samples = (filtered_samples * 32767).astype(np.int16)
    
    # Create a temporary enhanced file
    enhanced_file = input_file.replace('.mp3', '_enhanced.wav')
    enhanced_audio = AudioSegment(
        filtered_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    enhanced_audio.export(enhanced_file, format="wav")
    print(f"Enhanced audio saved to {enhanced_file}")
    
    # Use the enhanced file for diarization
    diarization_input = enhanced_file
    print(f"Using enhanced audio for diarization: {diarization_input}")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    print("Continuing with original audio file")
    diarization_input = input_file

# Try with 4 speakers for better dialect separation and use ProgressHook
try:
    # Using 4 speakers can help force the model to find more subtle differences
    print("Attempting diarization with num_speakers=4...")
    # Use ProgressHook to show real-time progress
    with ProgressHook() as hook:
        diarization = pipeline(diarization_input, num_speakers=4, hook=hook)
    print("Successfully ran diarization with num_speakers=4")
except Exception as e1:
    print(f"Error with num_speakers=4: {e1}")
    try:
        # Fall back to 3 speakers if 4 doesn't work
        print("Falling back to num_speakers=3...")
        with ProgressHook() as hook:
            diarization = pipeline(diarization_input, num_speakers=3, hook=hook)
        print("Successfully ran diarization with num_speakers=3")
    except Exception as e2:
        print(f"Error with num_speakers=3: {e2}")
        try:
            # Last resort - 2 speakers
            print("Falling back to num_speakers=2...")
            with ProgressHook() as hook:
                diarization = pipeline(diarization_input, num_speakers=2, hook=hook)
            print("Successfully ran diarization with num_speakers=2")
        except Exception as e3:
            print(f"Error with num_speakers=2: {e3}")
            # If all else fails, raise the original error
            raise e1

# Note: For better dialect separation, we're relying on the audio normalization
# step to help the algorithm distinguish between speakers with similar voices

# Process results
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

# Print timing information
end_time = time.time()
duration = end_time - start_time
print(f"\nDiarization completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
print(f"Found {len(segments)} speaker segments")

# Print and save segments
with open(output_file, "w") as f:
    for seg in segments:
        line = f"Speaker {seg['speaker']} from {seg['start']:.2f}s to {seg['end']:.2f}s"
        print(line)
        f.write(line + "\n")

print(f"\nSegments saved to {output_file}")
print(f"You can now use this file with fast-whisper-minimal-segments.py:")
print(f"python fast-whisper-minimal-segments.py {input_file} {output_file}")