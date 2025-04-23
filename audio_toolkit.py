#!/usr/bin/env python3
# Audio Toolkit - Main command-line interface
# Handles audio preprocessing, diarization, and SRT creation
# 2025-04-23 -JS

__version__ = "0.0.026"  # Version should match CHANGELOG.md

import os
import sys
import logging
import argparse
import datetime
import warnings
import torch

# Filter out warnings from dependencies
warnings.filterwarnings("ignore", message="torchaudio._backend.*has been deprecated")
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
warnings.filterwarnings("ignore", message="'audioop' is deprecated and slated for removal")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

# Specifically filter out the pydub audioop warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydub.utils")

# Filter out specific remaining pyannote warnings
warnings.filterwarnings("ignore", message="The get_cmap function was deprecated in Matplotlib 3.7")
warnings.filterwarnings("ignore", message="`torchaudio.backend.common.AudioMetaData` has been moved to `torchaudio.AudioMetaData`")
from pathlib import Path

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.audio_processing.preprocessor import AudioPreprocessor


def log(level, *messages, **kwargs):
    """
    Unified logging function.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        messages: Messages to log
        kwargs: Additional logging parameters
    """
    logger = logging.getLogger(__name__)
    
    if level == logging.DEBUG:
        logger.debug(*messages, **kwargs)
    elif level == logging.INFO:
        logger.info(*messages, **kwargs)
    elif level == logging.WARNING:
        logger.warning(*messages, **kwargs)
    elif level == logging.ERROR:
        logger.error(*messages, **kwargs)
    elif level == logging.CRITICAL:
        logger.critical(*messages, **kwargs)


def setup_logging(args):
    """
    Set up logging based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(logs_dir, f'audio-toolkit-{timestamp}.log')
    
    # Determine log level
    if args.debug:
        console_level = logging.DEBUG
        file_level = logging.DEBUG
    elif args.quiet:
        console_level = logging.WARNING
        file_level = logging.INFO
    else:
        console_level = logging.INFO
        file_level = logging.DEBUG
    
    # Configure logging
    logging.basicConfig(
        level=min(console_level, file_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if not args.quiet or args.debug else logging.NullHandler()
        ]
    )
    
    log(logging.INFO, f"Logging initialized. Log file: {log_file}")


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Audio Toolkit - Process, diarize, and create SRT files from audio recordings'
    )
    
    parser.add_argument(
        '--input-audio',
        required=True,
        help='Path to input audio file'
    )
    
    parser.add_argument(
        '--output-dir',
        default=os.getcwd(),
        help='Directory to save output files (default: current directory)'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip audio preprocessing steps'
    )
    
    parser.add_argument(
        '--skip-diarization',
        action='store_true',
        help='Skip speaker diarization step'
    )
    
    # Diarization parameters
    parser.add_argument(
        '--min-speakers',
        type=int,
        default=2,
        help='Minimum number of speakers to consider (default: 2)'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=4,
        help='Maximum number of speakers to consider (default: 4)'
    )
    
    parser.add_argument(
        '--clustering-threshold',
        type=float,
        default=0.65,
        help='Clustering threshold for speaker separation (default: 0.65)'
    )
    
    # SRT generation parameters
    parser.add_argument(
        '--skip-srt',
        action='store_true',
        help='Skip SRT subtitle file generation'
    )
    
    parser.add_argument(
        '--include-timestamps',
        action='store_true',
        help='Include timestamps in the SRT subtitle text'
    )
    
    parser.add_argument(
        '--speaker-format',
        type=str,
        default="{speaker}:",
        help='Format string for speaker labels in SRT file (default: "{speaker}:"'
    )
    
    parser.add_argument(
        '--max-gap',
        type=float,
        default=1.0,
        help='Maximum gap in seconds between segments to merge (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-duration',
        type=float,
        default=10.0,
        help='Maximum duration in seconds for a merged segment (default: 10.0)'
    )
    
    # WAV conversion parameters
    parser.add_argument(
        '--bit-depth',
        type=int,
        default=24,
        choices=[16, 24, 32],
        help='Bit depth for WAV conversion (default: 24)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        choices=[44100, 48000, 96000],
        help='Sample rate in Hz for WAV conversion (default: 48000)'
    )
    
    # Audio processing parameters
    parser.add_argument(
        '--highpass',
        type=int,
        default=3750,
        help='High-pass filter cutoff frequency in Hz (default: 3750, optimal for Swedish dialect isolation)'
    )
    
    parser.add_argument(
        '--lowpass',
        type=int,
        default=8000,
        help='Low-pass filter cutoff frequency in Hz (default: 8000)'
    )
    
    parser.add_argument(
        '--compression-threshold',
        type=float,
        default=-10.0,
        help='Compression threshold in dB (default: -10.0)'
    )
    
    parser.add_argument(
        '--compression-ratio',
        type=float,
        default=2.0,
        help='Compression ratio (default: 2.0)'
    )
    
    parser.add_argument(
        '--volume-gain',
        type=float,
        default=3.0,
        help='Volume gain in dB (default: 3.0)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'Audio Toolkit v{__version__}',
        help='Show version information and exit'
    )
    
    return parser.parse_args()


def process_audio(args):
    """
    Process audio file according to command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Get input file path
    input_file = os.path.abspath(args.input_audio)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get input filename without extension
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create run subdirectory with timestamp
    run_dir = os.path.join(output_dir, f"{timestamp}_{input_basename}")
    os.makedirs(run_dir, exist_ok=True)
    log(logging.INFO, f"Created output directory: {run_dir}")
    
    # Generate output file name based on input file
    output_basename = os.path.splitext(os.path.basename(input_file))[0] + "_processed.wav"
    output_file = os.path.join(run_dir, output_basename)
    
    # Create debug directory if debug mode is enabled
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(run_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        log(logging.INFO, f"Debug mode enabled, intermediate files will be saved to {debug_dir}")
    
    # Configure audio preprocessor
    config = {
        'debug': args.debug,
        'debug_dir': debug_dir,
        # WAV conversion parameters
        'bit_depth': args.bit_depth,
        'sample_rate': args.sample_rate,
        # Audio processing parameters
        'highpass_cutoff': args.highpass,
        'lowpass_cutoff': args.lowpass,
        'compression_threshold': args.compression_threshold,
        'compression_ratio': args.compression_ratio,
        'volume_gain': args.volume_gain
    }
    
    log(logging.INFO, f"Processing audio file: {input_file}")
    log(logging.INFO, f"Output will be saved to: {output_file}")
    log(logging.INFO, f"Using {config['bit_depth']}-bit depth and {config['sample_rate']}Hz sample rate")
    
    # Import here to ensure the mock in tests works correctly
    from src.audio_processing.preprocessor import AudioPreprocessor
    preprocessor = AudioPreprocessor(config)
    
    # Preprocess audio
    preprocessing_result = True
    if args.skip_preprocessing:
        log(logging.INFO, "Skipping preprocessing as requested")
    else:
        preprocessing_result = preprocessor.preprocess(input_file, output_file)
        if not preprocessing_result:
            log(logging.ERROR, "Preprocessing failed")
            return False
        input_file = output_file  # Use processed file for diarization
    
    # Perform speaker diarization unless explicitly skipped
    if not args.skip_diarization:
        log(logging.INFO, "Starting speaker diarization")
        
        # Configure diarizer
        diarization_config = {
            'debug': args.debug,
            'debug_dir': debug_dir if args.debug else None,
            'min_speakers': args.min_speakers,
            'max_speakers': args.max_speakers,
            'clustering_threshold': args.clustering_threshold,
            'use_gpu': torch.cuda.is_available(),
            'huggingface_token': os.environ.get('HF_TOKEN'),
            'batch_size': 32
        }
        
        # Import diarizer here to ensure the mock in tests works correctly
        from src.audio_processing.diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(diarization_config)
        
        # Run diarization
        diarization_segments = diarizer.diarize(input_file, run_dir)
        if not diarization_segments:
            log(logging.ERROR, "Diarization failed")
            return False
        
        log(logging.INFO, "Speaker diarization completed successfully")
        
        # Generate SRT file unless explicitly skipped
        if not args.skip_srt and diarization_segments:
            log(logging.INFO, "Generating SRT subtitle file")
            
            # Get diarization result
            diarization_result = diarizer.get_diarization_result()
            
            # Create SRT generator
            srt_config = {
                'include_timestamps': args.include_timestamps,
                'speaker_format': args.speaker_format,
                'max_gap': args.max_gap,
                'max_duration': args.max_duration
            }
            
            log(logging.INFO, f"SRT configuration: include_timestamps={args.include_timestamps}, speaker_format='{args.speaker_format}', max_gap={args.max_gap}s, max_duration={args.max_duration}s")
            
            from src.audio_processing.srt_generator import SRTGenerator
            srt_generator = SRTGenerator(srt_config)
            
            # Merge segments if needed
            log(logging.INFO, f"Merging segments (max_gap={args.max_gap}s, max_duration={args.max_duration}s)...")
            merged_segments = srt_generator.merge_segments(
                diarization_result,
                max_gap=args.max_gap,
                max_duration=args.max_duration
            )
            
            # Generate SRT file
            log(logging.INFO, "Generating SRT file...")
            srt_file = os.path.join(run_dir, os.path.splitext(input_basename)[0] + ".srt")
            if not srt_generator.generate_srt(
                merged_segments, 
                srt_file,
                speaker_format=args.speaker_format,
                include_timestamps=args.include_timestamps
            ):
                log(logging.ERROR, "SRT generation failed")
                return False
            
            log(logging.INFO, f"SRT file generated successfully: {srt_file}")
            log(logging.INFO, "SRT generation completed")
    
    return True


def check_dependencies():
    """
    Check that all required dependencies are installed.
    Provides helpful error messages if missing dependencies directly to console.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    # Check for FFmpeg
    try:
        # Try to import torio which requires FFmpeg
        import torio
        log(logging.DEBUG, "Successfully imported torio")
    except ImportError as e:
        error_msg = f"Failed to import torio: {str(e)}"
        print(f"\033[91mERROR: {error_msg}\033[0m")  # Red text for error
        log(logging.ERROR, error_msg)
        missing_deps.append("torio")
    
    # Check for FFmpeg libraries specifically
    try:
        import ctypes
        try:
            # Try to find FFmpeg libraries with different version numbers
            # First check if ffmpeg command is available
            import subprocess
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                log(logging.DEBUG, "FFmpeg command is available")
                
                # Try to load common versions of libavutil
                ffmpeg_found = False
                for lib_version in ["libavutil.so", "libavutil.so.58", "libavutil.so.57", "libavutil.so.56"]:
                    try:
                        ctypes.CDLL(lib_version)
                        ffmpeg_found = True
                        log(logging.DEBUG, f"Found FFmpeg library: {lib_version}")
                        break
                    except OSError:
                        continue
                
                if ffmpeg_found:
                    log(logging.DEBUG, "FFmpeg libraries found")
                else:
                    warning_msg = "FFmpeg command is available but libraries not found in standard locations"
                    print(f"\033[93mWARNING: {warning_msg}\033[0m")  # Yellow text for warning
                    print("\033[93mWARNING: This might cause issues with some audio processing functions\033[0m")
                    log(logging.WARNING, warning_msg)
                    log(logging.WARNING, "This might cause issues with some audio processing functions")
            except (subprocess.SubprocessError, FileNotFoundError):
                error_msg = "FFmpeg command not found. Please install FFmpeg with:"
                print(f"\033[91mERROR: {error_msg}\033[0m")
                print("\033[91m       sudo apt-get install ffmpeg libavutil-dev libavcodec-dev libavformat-dev\033[0m")
                log(logging.ERROR, error_msg)
                log(logging.ERROR, "sudo apt-get install ffmpeg libavutil-dev libavcodec-dev libavformat-dev")
                missing_deps.append("ffmpeg")
        except OSError:
            error_msg = "Missing FFmpeg libraries. Please install them with:"
            print(f"\033[91mERROR: {error_msg}\033[0m")
            print("\033[91m       sudo apt-get install ffmpeg libavutil-dev libavcodec-dev libavformat-dev\033[0m")
            log(logging.ERROR, error_msg)
            log(logging.ERROR, "sudo apt-get install ffmpeg libavutil-dev libavcodec-dev libavformat-dev")
            missing_deps.append("ffmpeg-libs")
    except ImportError:
        warning_msg = "Could not check for FFmpeg libraries (ctypes not available)"
        print(f"\033[93mWARNING: {warning_msg}\033[0m")
        log(logging.WARNING, warning_msg)
    
    if missing_deps:
        error_msg = f"Missing dependencies: {', '.join(missing_deps)}"
        print(f"\033[91mERROR: {error_msg}\033[0m")
        print("\033[91mERROR: Please install the missing dependencies and try again\033[0m")
        log(logging.ERROR, error_msg)
        log(logging.ERROR, "Please install the missing dependencies and try again")
        return False
    
    return True

def main():
    """
    Main entry point for the audio toolkit.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args)
    
    log(logging.INFO, "Starting Audio Toolkit")
    
    # Check dependencies before proceeding
    if not check_dependencies():
        print("\033[91m\nDependency check failed. Please see the FAQ.md file for detailed solutions.\033[0m")
        log(logging.ERROR, "Dependency check failed. Please install the missing dependencies.")
        sys.exit(1)
    
    try:
        # Process audio
        if process_audio(args):
            log(logging.INFO, "Audio processing completed successfully")
            sys.exit(0)
        else:
            log(logging.ERROR, "Audio processing failed")
            sys.exit(1)
            
    except Exception as e:
        log(logging.CRITICAL, f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
