#!/usr/bin/env python3
"""
Advanced diarization script that uses separate models for segmentation and clustering.
This approach is better for distinguishing between similar voices like Swedish dialects.
"""

# Import libraries
import os
import sys
import time
import argparse
import warnings
import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Import pydub for audio processing
from pydub import AudioSegment
from scipy import signal

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Specifically filter torchaudio and speechbrain warnings
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*speechbrain.*")
warnings.filterwarnings("ignore", message=".*AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*backend.*")

# Enable TF32 for faster processing (as requested by user)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Get input filename from command line argument or use default
input_file = sys.argv[1] if len(sys.argv) > 1 else "vocals_clean.mp3"

# Generate output filename based on input filename
output_file = os.path.splitext(input_file)[0] + ".segments"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Processing file: {input_file}")
print(f"Output will be saved to: {output_file}")

# Start timing
start_time = time.time()

# Optimize GPU usage
if torch.cuda.is_available():
    print("Optimizing GPU usage...")
    # Enable TF32 for faster processing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set benchmark mode for faster processing with fixed input sizes
    torch.backends.cudnn.benchmark = True
    # Try to allocate more GPU memory
    try:
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.95)
        print("Reserved 95% of GPU memory for processing")
    except Exception as e:
        print(f"Could not optimize GPU memory: {e}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load models for diarization
print("Loading diarization models...")

# Load separate models for segmentation and diarization
print("\n=== Loading separate models for segmentation and diarization ===")

# Define the models to use for each step
segmentation_models = [
    "pyannote/segmentation-3.0"  # Using the newer 3.0 version
]

diarization_models = [
    "tensorlake/speaker-diarization-3.1",  # Newer model that might handle dialects better
    "pyannote/speaker-diarization-3.1"    # Standard pyannote 3.1 model

]


vad_models = [
    "pyannote/voice-activity-detection"
]

# Speaker counts to try
speaker_counts = [4, 3, 2]  # Try different speaker counts

# Properly authenticate and load all models
# The HuggingFace token should work for all models if you have access to them

# Load Voice Activity Detection (VAD) model
vad_pipeline = None
try:
    print(f"Loading VAD model: {vad_models[0]}")
    vad_pipeline = Pipeline.from_pretrained(
        vad_models[0],
        use_auth_token="[TOKEN]"
    )
    if torch.cuda.is_available():
        vad_pipeline.to(torch.device("cuda"))
    print(f"Successfully loaded VAD model: {vad_models[0]}")
except Exception as e:
    print(f"Error loading VAD model: {e}")
    print("You may need to accept the user conditions at https://hf.co/pyannote/voice-activity-detection")
    print("Continuing without VAD model")
    vad_pipeline = None

# Load Segmentation model for Voice Activity Detection
segmentation_model = None
vad_pipeline = None
try:
    print(f"Loading segmentation model: {segmentation_models[0]}")
    # Load the segmentation model
    from pyannote.audio import Model
    segmentation_model = Model.from_pretrained(
        segmentation_models[0],
        use_auth_token="[TOKEN]"
    )
    
    if torch.cuda.is_available():
        segmentation_model.to(torch.device("cuda"))
    
    # Create a Voice Activity Detection pipeline using the segmentation model
    from pyannote.audio.pipelines import VoiceActivityDetection
    vad_pipeline = VoiceActivityDetection(segmentation=segmentation_model)
    
    # Set parameters
    hyper_parameters = {
        # remove speech regions shorter than that many seconds
        "min_duration_on": 0.1,
        # fill non-speech regions shorter than that many seconds
        "min_duration_off": 0.1
    }
    
    vad_pipeline.instantiate(hyper_parameters)
    print(f"Successfully loaded segmentation model and created VAD pipeline")
except Exception as e:
    print(f"Error loading segmentation model: {e}")
    print("Continuing without segmentation model - will use diarization model directly")
    segmentation_model = None
    vad_pipeline = None

# Load Diarization model
diarization_pipeline = None
for model_name in diarization_models:
    try:
        print(f"Loading diarization model: {model_name}")
        diarization_pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token="[TOKEN]"
        )
        if torch.cuda.is_available():
            diarization_pipeline.to(torch.device("cuda"))
        print(f"Successfully loaded diarization model: {model_name}")
        
        # Set diarization parameters
        if hasattr(diarization_pipeline, "min_speakers"):
            diarization_pipeline.min_speakers = 2
            print("Set min_speakers to 2")
        if hasattr(diarization_pipeline, "max_speakers"):
            diarization_pipeline.max_speakers = 4
            print("Set max_speakers to 4")
        if hasattr(diarization_pipeline, "clustering_threshold"):
            diarization_pipeline.clustering_threshold = 0.65  # Lower threshold to separate similar voices
            print("Set clustering_threshold to 0.65")
        
        break
    except Exception as e:
        print(f"Error loading diarization model {model_name}: {e}")

# Check if we have all the required models
if not vad_pipeline and not diarization_pipeline:
    print("Failed to load any models. Exiting.")
    sys.exit(1)

# Track which models we're using
using_separate_models = vad_pipeline is not None and diarization_pipeline is not None
if using_separate_models:
    print("\nSuccessfully loaded separate models for each step of the diarization process")
else:
    print("\nCould not load all separate models, will use available models")
    
    # If we don't have a diarization model, we can't proceed
    if not diarization_pipeline:
        print("No diarization model available. Exiting.")
        sys.exit(1)

# Initialize tracking variables for the multi-stage process
successful_runs = []

# Simplified audio preprocessing to avoid pydub issues
print("Checking audio file...")

# Just use the original file directly
diarization_input = input_file
print(f"Using original audio file for diarization: {diarization_input}")

# Verify the file exists
if not os.path.exists(diarization_input):
    print(f"Error: Audio file {diarization_input} not found")
    sys.exit(1)
else:
    print(f"Audio file {diarization_input} found and ready for processing")

# Run a simplified diarization process with different speaker counts
print("\n=== Running optimized diarization process ===\n")

# Initialize results storage
all_results = {}

# Process the audio using the diarization model with different speaker counts
try:
    # First run Voice Activity Detection if available
    speech_regions = None
    if vad_pipeline is not None:
        print("\n--- Running Voice Activity Detection first ---")
        vad_start_time = time.time()
        
        # Run VAD to get speech regions
        vad_result = vad_pipeline(diarization_input)
        
        # Extract speech regions
        speech_regions = []
        for speech, _, _ in vad_result.itertracks(yield_label=True):
            speech_regions.append({
                "start": speech.start,
                "end": speech.end
            })
        
        vad_end_time = time.time()
        print(f"Detected {len(speech_regions)} speech regions in {vad_end_time - vad_start_time:.2f} seconds")
        
        # Save VAD results to file
        vad_output_file = os.path.splitext(input_file)[0] + ".vad.segments"
        with open(vad_output_file, "w") as f:
            for seg in speech_regions:
                line = f"Speech from {seg['start']:.2f}s to {seg['end']:.2f}s"
                f.write(line + "\n")
        print(f"Voice activity detection results saved to {vad_output_file}")
    
    # Try different speaker counts
    best_speaker_count = None
    max_segments = 0
    
    for num_speakers in speaker_counts:
        try:
            print(f"\n--- Trying with num_speakers={num_speakers} ---")
            start_time_run = time.time()
            
            # Run diarization with the current speaker count
            with ProgressHook() as hook:
                diarization = diarization_pipeline(diarization_input, num_speakers=num_speakers, hook=hook)
            
            # Process results for this run
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Generate output filename for this speaker count
            run_output_file = os.path.splitext(input_file)[0] + f".{num_speakers}speakers.segments"
            
            # Save segments to file
            with open(run_output_file, "w") as f:
                for seg in segments:
                    line = f"Speaker {seg['speaker']} from {seg['start']:.2f}s to {seg['end']:.2f}s"
                    f.write(line + "\n")
            
            # Calculate duration and stats
            end_time_run = time.time()
            duration_run = end_time_run - start_time_run
            
            # Store results
            all_results[f"{num_speakers}speakers"] = {
                "segments": segments,
                "output_file": run_output_file,
                "duration": duration_run,
                "segment_count": len(segments),
                "num_speakers": num_speakers
            }
            
            # Check if this is the best run so far
            if len(segments) > max_segments:
                max_segments = len(segments)
                best_speaker_count = num_speakers
            
            print(f"Successfully completed diarization with {num_speakers} speakers")
            print(f"Found {len(segments)} speaker segments in {duration_run:.2f} seconds")
            print(f"Results saved to {run_output_file}")
            
            # Add to successful runs
            successful_runs.append(num_speakers)
            
        except Exception as e:
            print(f"Error with num_speakers={num_speakers}: {e}")
    
    # If we have no successful runs, try without specifying speaker count
    if not successful_runs:
        try:
            print("\n--- Trying with auto speaker detection ---")
            start_time_run = time.time()
            
            with ProgressHook() as hook:
                diarization = diarization_pipeline(diarization_input, hook=hook)
            
            # Process results
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Generate output filename
            run_output_file = os.path.splitext(input_file)[0] + ".auto.segments"
            
            # Save segments to file
            with open(run_output_file, "w") as f:
                for seg in segments:
                    line = f"Speaker {seg['speaker']} from {seg['start']:.2f}s to {seg['end']:.2f}s"
                    f.write(line + "\n")
            
            # Calculate duration and stats
            end_time_run = time.time()
            duration_run = end_time_run - start_time_run
            
            # Store results
            all_results["auto"] = {
                "segments": segments,
                "output_file": run_output_file,
                "duration": duration_run,
                "segment_count": len(segments),
                "num_speakers": "auto"
            }
            
            best_speaker_count = "auto"
            
            print(f"Successfully completed diarization with auto speaker detection")
            print(f"Found {len(segments)} speaker segments in {duration_run:.2f} seconds")
            print(f"Results saved to {run_output_file}")
            
            successful_runs.append("auto")
            
        except Exception as e:
            print(f"Error with auto speaker detection: {e}")
    
    # Create a summary of all runs
    print("\n=== Diarization Results Summary ===\n")
    print(f"Total successful runs: {len(all_results)}")
    
    if best_speaker_count is not None:
        best_run_id = f"{best_speaker_count}speakers" if best_speaker_count != "auto" else "auto"
        best_result = all_results[best_run_id]
        
        print(f"\nRecommended configuration (most speaker segments):\n")
        print(f"Speaker count: {best_result['num_speakers']}")
        print(f"Segments: {best_result['segment_count']}")
        print(f"Output file: {best_result['output_file']}")
        print(f"\nTo use this file with transcription:")
        print(f"python fast-whisper-minimal-segments.py {input_file} {best_result['output_file']}")
        
        # Set the output segments for the final processing
        segments = best_result["segments"]
        output_file = best_result["output_file"]
    else:
        print("No successful diarization runs.")
        sys.exit(1)
    
    print("\nAll available segment files:")
    for run_id, result in all_results.items():
        print(f"- {result['output_file']} ({result['num_speakers']} speakers, {result['segment_count']} segments)")
    
except Exception as e:
    print(f"Error during diarization: {e}")
    sys.exit(1)

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
