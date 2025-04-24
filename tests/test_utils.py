#!/usr/bin/env python3
# Test utilities for the audio-to-srt-toolkit
# Contains shared helper functions for tests
# 2025-04-24 -JS

def create_test_args(input_file, output_dir):
    """
    Create a test Args object with all required parameters.
    This ensures tests don't break when new parameters are added.
    
    Args:
        input_file: Path to the input audio file
        output_dir: Path to the output directory
        
    Returns:
        Args: An Args object with all required parameters
    
    2025-04-24 -JS
    """
    class Args:
        pass
        
    args = Args()
    
    # Basic parameters
    args.input_audio = input_file
    args.output_dir = output_dir
    args.skip_preprocessing = False
    args.skip_diarization = False
    args.skip_srt = False
    args.skip_transcription = False
    
    # Audio processing parameters
    args.highpass = 150
    args.lowpass = 8000
    args.compression_threshold = -20
    args.compression_ratio = 4.0
    args.volume_gain = 3.0
    args.bit_depth = 24
    args.sample_rate = 48000
    
    # New options for skipping steps
    args.use_vocals_directly = False
    args.skip_steps = None
    args.list_steps = False
    
    # Diarization parameters
    args.speaker_count = None
    args.min_speakers = 2
    args.max_speakers = 4
    args.clustering_threshold = 0.65
    
    # SRT parameters
    args.generate_srt = True
    args.include_timestamps = False
    args.speaker_format = "{speaker}:"
    args.max_gap = 1.0
    args.max_duration = 10.0
    args.srt_pre = 0.1
    args.srt_post = 0.1
    args.srt_min_duration = 0.3
    args.srt_no_speaker = False
    args.confidence_threshold = 0.5
    args.max_segments = 0
    
    # Continuation parameters
    args.continue_from = None
    args.continue_folder = None
    
    # Debug parameters
    args.quiet = False
    args.debug = False
    args.debug_files_only = False
    
    return args
