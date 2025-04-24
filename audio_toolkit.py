#!/usr/bin/env python3
# Audio Toolkit - Main command-line interface
# Handles audio preprocessing, diarization, and SRT creation
# 2025-04-23 -JS

__version__ = "0.0.054"  # Version should match CHANGELOG.md

import os
import sys
import logging
import argparse
import datetime
import warnings
import subprocess
import yaml
import torch
import re  # 2025-04-24 -JS
import glob  # 2025-04-24 -JS

def setup_warning_filters():
    """
    Set up warning filters to suppress known dependency warnings.
    This function centralizes all warning filtering in one place for better maintainability.
    
    Filters warnings from:
    - torchaudio deprecation warnings
    - speechbrain module deprecation warnings
    - audioop deprecation in pydub
    - matplotlib deprecation warnings
    - pyannote-specific warnings
    
    2025-04-24 -JS
    """
    # Reset all filters first to ensure our filters take precedence
    warnings.resetwarnings()
    
    # Filter out warnings from torchaudio - use exact message patterns
    warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")
    warnings.filterwarnings("ignore", message="torchaudio._backend.get_audio_backend has been deprecated")
    warnings.filterwarnings("ignore", message="`torchaudio.backend.common.AudioMetaData` has been moved to `torchaudio.AudioMetaData`")
    
    # Filter out warnings from speechbrain
    warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
    
    # Filter out audioop deprecation warnings
    warnings.filterwarnings("ignore", message="'audioop' is deprecated and slated for removal")
    
    # Filter out matplotlib deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
    warnings.filterwarnings("ignore", message="The get_cmap function was deprecated in Matplotlib 3.7")
    
    # Filter out pydub warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydub.utils")
    
    # Filter out NumExpr/TensorFlow warnings from pyannote
    # 2025-04-24 -JS
    warnings.filterwarnings("ignore", message="NumExpr detected 8 cores")
    warnings.filterwarnings("ignore", message="TensorFloat-32 (TF32) has been disabled")
    warnings.filterwarnings("ignore", message="TensorFloat-32*")
    
    # Add more specific filters for common warnings
    warnings.filterwarnings("ignore", message=".*ffmpeg/avlib.*")
    warnings.filterwarnings("ignore", message=".*Applied quirks.*")
    warnings.filterwarnings("ignore", message=".*Excluded quirks specified by the.*")

# Set up warning filters at import time
setup_warning_filters()
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


# Custom filter to suppress progress bar output in logs
class ProgressBarFilter(logging.Filter):
    """Filter out progress bar output from logs."""
    def filter(self, record):
        # Check if the log message contains progress bar indicators
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            # Check for common progress bar patterns
            if '|' in msg and '%' in msg and any(char in msg for char in ['█', '▉', '▊', '▋', '▌', '▍', '▎', 'it/s', 's/it']):
                return False  # Filter out progress bars
        return True  # Keep all other messages


def setup_logging(args):
    """
    Set up logging based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log file name with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(logs_dir, f'audio-toolkit-{timestamp}.log')
    
    # Set up file logging level - always DEBUG with debug flags
    if args.debug or args.debug_files_only:
        file_log_level = logging.DEBUG
    elif args.quiet:
        file_log_level = logging.WARNING
    else:
        file_log_level = logging.INFO
    
    # Set up console logging level - DEBUG only with --debug flag
    if args.debug:
        console_log_level = logging.DEBUG
    elif args.quiet:
        console_log_level = logging.WARNING
    else:
        console_log_level = logging.INFO
    
    # Create formatters for different outputs
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')  # Simple format for console
    
    # Create file handler with progress bar filter
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(ProgressBarFilter())  # Add filter to suppress progress bars
    
    # Create console handler
    if args.quiet:
        # No console output if quiet
        console_handler = logging.NullHandler()
    else:
        # Normal console output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(ProgressBarFilter())  # Add filter to console output
    
    # Configure logging - use DEBUG level to allow all handlers to work
    logging.basicConfig(
        level=logging.DEBUG,  # Base level is DEBUG to allow handlers to control their own levels
        handlers=[file_handler, console_handler]
    )
    
    log(logging.DEBUG, f"Logging initialized. Log file: {log_file}")


def parse_args(args=None):
    # args parameter allows for testing with custom arguments
    # 2025-04-24 -JS
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Audio Toolkit - Process, diarize, and create SRT files from audio recordings'
    )
    
    # 2025-04-24 -JS
    # Create a mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # Add input-audio to the group
    input_group.add_argument(
        '--input-audio',
        help='Path to input audio file'
    )
    
    # Add continuation to the group as a single option
    input_group.add_argument(
        '--continue-folder',
        help='Output folder from a previous run to continue from'
    )
    
    # Add continue-from as a separate argument
    parser.add_argument(
        '--continue-from',
        choices=['preprocessing', 'diarization', 'srt'],
        help='Continue processing from a specific step (requires --continue-folder)'
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
    
    # Transcription parameters
    # 2025-04-24 -JS
    parser.add_argument(
        '--max-segments',
        type=int,
        help='Maximum number of segments to transcribe (for testing)'
    )
    
    parser.add_argument(
        '--skip-transcription',
        action='store_true',
        help='Skip transcription and use placeholders'
    )
    
    # Segment padding for transcription
    # 2025-04-24 -JS
    parser.add_argument(
        '--srt-pre',
        type=float,
        default=0.1,
        help='Seconds to add before each segment for transcription (default: 0.1)'
    )
    
    parser.add_argument(
        '--srt-post',
        type=float,
        default=0.1,
        help='Seconds to add after each segment for transcription (default: 0.1)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for transcriptions (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--srt-min-duration',
        type=float,
        default=0.3,
        help='Minimum segment duration in seconds to include in SRT (default: 0.3)'
    )
    
    parser.add_argument(
        '--srt-no-speaker',
        action='store_true',
        help='Remove speaker labels from SRT output'
    )
    
    parser.add_argument(
        '--srt-remove-empty',
        action='store_true',
        help='Remove empty segments from SRT output even with --srt-min-duration 0'
    )
    
    parser.add_argument(
        '--srt-empty-placeholder',
        type=str,
        default='[UNRECOGNIZABLE]',
        help='Text to use for empty segments (default: "[UNRECOGNIZABLE]")'
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
        '--use-vocals-directly',
        action='store_true',
        help='Use the vocals file directly for transcription, skipping all post-processing steps'
    )
    
    parser.add_argument(
        '--skip-steps',
        type=str,
        help='Comma-separated list of processing steps to skip (e.g., "highpass,lowpass")'
    )
    
    parser.add_argument(
        '--list-steps',
        action='store_true',
        help='List all available processing steps that can be skipped'
    )
    
    parser.add_argument(
        '--highpass',
        type=int,
        default=300,
        help='High-pass filter cutoff frequency in Hz (default: 300)'
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
        default=12.0,
        help='Volume gain in dB (default: 12.0)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging to both console and files'
    )
    
    parser.add_argument(
        '--debug-files-only',
        action='store_true',
        help='Create debug files in output_dir/debug/ while keeping console at INFO level'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'Audio Toolkit v{__version__}',
        help='Show version information and exit'
    )
    
    # Speaker count option
    # 2025-04-24 -JS
    
    parser.add_argument(
        '--speaker-count',
        type=int,
        help='Specify the number of speakers for diarization (overrides config)'
    )
    
    # Use provided args for testing or default to sys.argv
    # 2025-04-24 -JS
    args = parser.parse_args(args)
    
    # 2025-04-24 -JS
    # Validate continuation requirements
    if args.continue_folder and not args.continue_from:
        parser.error("--continue-from is required when using --continue-folder")
    
    return args


def load_config():
    """
    Load configuration from config.yaml file.
    
    Returns:
        dict: Configuration dictionary or empty dict if file not found/invalid
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            log(logging.INFO, f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            log(logging.WARNING, f"Error loading config file: {str(e)}")
    else:
        log(logging.WARNING, f"Config file not found at {config_path}")
    
    # Return empty config if file not found or error occurred
    return {}

def process_audio(args):
    """
    Process audio file according to command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we're continuing from a previous run
    # 2025-04-24 -JS
    if args.continue_from and args.continue_folder:
        run_dir = os.path.abspath(args.continue_folder)
        if not os.path.exists(run_dir):
            log(logging.ERROR, f"Continuation folder does not exist: {run_dir}")
            return False
        log(logging.INFO, f"Continuing from previous run in: {run_dir}")
        
        # When continuing from SRT, we don't need the original input file at all
        # 2025-04-24 -JS
        input_file = None
        
        # Only look for the original input file if we're not continuing from SRT
        if args.continue_from != 'srt':
            # Try to find the original input audio file from version_info.txt
            version_info_path = os.path.join(run_dir, "version_info.txt")
            
            if os.path.exists(version_info_path):
                try:
                    with open(version_info_path, 'r') as f:
                        content = f.read()
                        
                    # Try to find the input file from different formats
                    # First check for the new format (Input File: path)
                    input_file_match = re.search(r"Input File:\s*(.+)\n", content)
                    if input_file_match:
                        input_file = input_file_match.group(1).strip()
                    
                    # If not found, check for the older format (- File: path)
                    if not input_file:
                        file_match = re.search(r"- File:\s*(.+)\n", content)
                        if file_match:
                            input_file = file_match.group(1).strip()
                    
                    # Check if the file exists
                    if input_file:
                        if os.path.exists(input_file):
                            log(logging.INFO, f"Retrieved original input file: {input_file}")
                        else:
                            log(logging.WARNING, f"Original input file not found at: {input_file}")
                            log(logging.INFO, "Will look for audio files in the output folder instead")
                            input_file = None
                    else:
                        log(logging.WARNING, "Could not find original input file information")
                        input_file = None
                except Exception as e:
                    log(logging.WARNING, f"Could not read version_info.txt: {str(e)}")
        
        # Always check if the input file exists, regardless of where we got it from
        # 2025-04-24 -JS
        if input_file and not os.path.exists(input_file):
            log(logging.WARNING, f"Input file does not exist: {input_file}")
            input_file = None
            
        # If we don't have a valid input file, look for audio files in the folder
        if not input_file:
            # Look for audio files in the folder
            audio_files = []
            for ext in [".wav", ".mp3", ".mp4", ".m4a", ".flac"]:
                audio_files.extend(glob.glob(os.path.join(run_dir, f"*{ext}")))
            
            # When continuing from SRT, include both processed and original files
            # but prioritize processed WAV files in the display order
            # 2025-04-24 -JS
            if args.continue_from == 'srt':
                processed_wavs = [f for f in audio_files if f.endswith("_processed.wav")]
                other_audio = [f for f in audio_files if not f.endswith("_processed.wav")]
                # Reorder to show processed files first, but keep all files
                audio_files = processed_wavs + other_audio
            
            if len(audio_files) == 1:
                input_file = audio_files[0]
                log(logging.INFO, f"Found single audio file in folder: {input_file}")
            elif len(audio_files) > 1:
                log(logging.INFO, f"Found {len(audio_files)} audio files in folder. Please select one:")
                for i, file in enumerate(audio_files):
                    print(f"  [{i+1}] {os.path.basename(file)}")
                
                while True:
                    try:
                        choice = input("Enter the number of the audio file to use: ")
                        idx = int(choice) - 1
                        if 0 <= idx < len(audio_files):
                            input_file = audio_files[idx]
                            log(logging.INFO, f"Selected audio file: {input_file}")
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(audio_files)}")
                    except ValueError:
                        print("Please enter a valid number")
            
            # If we still don't have an input file, use the one provided or fail
            if not input_file:
                if not args.input_audio:
                    log(logging.ERROR, "Could not determine input audio file. Please provide --input-audio.")
                    return False
                input_file = os.path.abspath(args.input_audio)
                log(logging.INFO, f"Using provided input file: {input_file}")
        
        # Get input filename without extension for later use
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
    else:
        # Not continuing from a previous run, so we need an input file
        if not args.input_audio:
            log(logging.ERROR, "Input audio file is required when not continuing from a previous run")
            return False
            
        input_file = os.path.abspath(args.input_audio)
        
        # Get input filename without extension
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Generate timestamp for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run subdirectory with timestamp
        run_dir = os.path.join(output_dir, f"{timestamp}_{input_basename}")
        os.makedirs(run_dir, exist_ok=True)
        log(logging.INFO, f"Created output directory: {run_dir}")
    
    # Create version_info.txt file in the run directory with detailed information
    version_info_path = os.path.join(run_dir, "version_info.txt")
    
    # Get input audio file information
    try:
        from pydub import AudioSegment
        input_audio = AudioSegment.from_file(input_file)
        audio_info = {
            "duration_seconds": len(input_audio) / 1000,
            "duration_formatted": str(datetime.timedelta(milliseconds=len(input_audio))),
            "channels": input_audio.channels,
            "sample_width_bits": input_audio.sample_width * 8,
            "frame_rate": input_audio.frame_rate,
            "frame_count": int(len(input_audio) / 1000 * input_audio.frame_rate),
            "file_size_bytes": os.path.getsize(input_file),
            "file_size_mb": os.path.getsize(input_file) / (1024 * 1024)
        }
    except Exception as e:
        log(logging.WARNING, f"Could not get detailed input file info: {str(e)}")
        audio_info = {"error": f"Could not analyze input file: {str(e)}"}
    
    # Get system information
    import platform
    system_info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }
    
    # Write all information to the file
    with open(version_info_path, "w") as f:
        # Version and processing information
        f.write(f"Audio Toolkit Version: {__version__}\n")
        f.write(f"Processing Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Input File: {input_file}\n\n")
        
        # Input file information
        f.write("Input File Information:\n")
        f.write(f"- File: {input_file}\n")
        if "error" in audio_info:
            f.write(f"- {audio_info['error']}\n")
        else:
            f.write(f"- Duration: {audio_info['duration_formatted']} ({audio_info['duration_seconds']:.2f} seconds)\n")
            f.write(f"- Channels: {audio_info['channels']}\n")
            f.write(f"- Sample Width: {audio_info['sample_width_bits']} bits\n")
            f.write(f"- Sample Rate: {audio_info['frame_rate']} Hz\n")
            f.write(f"- Frame Count: {audio_info['frame_count']}\n")
            f.write(f"- File Size: {audio_info['file_size_mb']:.2f} MB ({audio_info['file_size_bytes']} bytes)\n\n")
        
        # Processing parameters
        f.write("Processing Parameters:\n")
        f.write(f"- Bit Depth: {args.bit_depth}\n")
        f.write(f"- Sample Rate: {args.sample_rate}\n")
        f.write(f"- Highpass Cutoff: {args.highpass} Hz\n")
        f.write(f"- Lowpass Cutoff: {args.lowpass} Hz\n")
        f.write(f"- Compression Threshold: {args.compression_threshold} dB\n")
        f.write(f"- Compression Ratio: {args.compression_ratio}:1\n")
        f.write(f"- Volume Gain: {args.volume_gain} dB\n\n")
        
        # Processing flags
        f.write("Processing Flags:\n")
        f.write(f"- Skip Preprocessing: {args.skip_preprocessing}\n")
        f.write(f"- Skip Diarization: {args.skip_diarization}\n")
        f.write(f"- Skip SRT Generation: {args.skip_srt}\n")
        f.write(f"- Debug Mode: {args.debug}\n\n")
        
        # System information
        f.write("System Information:\n")
        f.write(f"- Python Version: {system_info['python_version']}\n")
        f.write(f"- Operating System: {system_info['system']} {system_info['release']}\n")
        f.write(f"- Machine: {system_info['machine']}\n")
        f.write(f"- Processor: {system_info['processor']}\n")
    
    log(logging.DEBUG, f"Created detailed version info file: {version_info_path}")
    
    # Generate output file name based on input file
    output_basename = os.path.splitext(os.path.basename(input_file))[0] + "_processed.wav"
    output_file = os.path.join(run_dir, output_basename)
    
    # Create debug directory if debug mode or debug-files-only is enabled
    debug_dir = None
    if args.debug or args.debug_files_only:
        debug_dir = os.path.join(run_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        log(logging.DEBUG, f"Debug files will be saved to {debug_dir}")
        
    # Define output file path for preprocessed audio
    # 2025-04-24 -JS
    # Files inside the timestamped directory should not have timestamps themselves
    output_file = os.path.join(run_dir, f"{input_basename}_processed.wav")
    
    # Configure audio preprocessor
    config = {
        'debug': args.debug or args.debug_files_only,
        'debug_dir': debug_dir,
        # WAV conversion parameters
        'bit_depth': args.bit_depth,
        'sample_rate': args.sample_rate,
        # Audio processing parameters
        'highpass_cutoff': args.highpass,
        'lowpass_cutoff': args.lowpass,
        'compression_threshold': args.compression_threshold,
        'compression_ratio': args.compression_ratio,
        'volume_gain': args.volume_gain,
        # New options for skipping steps
        'use_vocals_directly': args.use_vocals_directly,
        'skip_steps': args.skip_steps.split(',') if args.skip_steps else []
    }
    
    log(logging.INFO, f"Processing audio file: {input_file}")
    log(logging.DEBUG, f"Output will be saved to: {output_file}")
    log(logging.DEBUG, f"Using {config['bit_depth']}-bit depth and {config['sample_rate']}Hz sample rate")
    
    # Import here to ensure the mock in tests works correctly
    from src.audio_processing.preprocessor import AudioPreprocessor
    preprocessor = AudioPreprocessor(config)
    
    # Preprocess audio
    preprocessing_result = True
    # Skip preprocessing if explicitly requested or if continuing from a later step
    # 2025-04-24 -JS
    if args.skip_preprocessing or (args.continue_from and args.continue_from != 'preprocessing'):
        if args.continue_from:
            log(logging.INFO, f"Skipping preprocessing due to continuation from {args.continue_from}")
        else:
            log(logging.INFO, "Skipping preprocessing as requested")
        
        # If continuing and the processed file exists, use it
        if args.continue_from and os.path.exists(output_file):
            input_file = output_file
    else:
        preprocessing_result = preprocessor.preprocess(input_file, output_file)
        if not preprocessing_result:
            log(logging.ERROR, "Preprocessing failed")
            return False
        input_file = output_file  # Use processed file for diarization
    
    # Perform speaker diarization unless explicitly skipped or continuing from SRT generation
    # 2025-04-24 -JS
    diarization_segments = None
    if not args.skip_diarization and not (args.continue_from and args.continue_from == 'srt'):
        log(logging.INFO, "Starting speaker diarization")
        
        # Load configuration from config file
        config_data = load_config()
        
        # Get Hugging Face token from config file or environment variable
        hf_token = None
        if config_data and 'authentication' in config_data and 'huggingface_token' in config_data['authentication']:
            hf_token = config_data['authentication']['huggingface_token']
            if hf_token == "your_token_here":
                log(logging.WARNING, "Default Hugging Face token found in config.yaml. Please update with your actual token.")
                hf_token = None
        
        # Fall back to environment variable if not in config
        if not hf_token:
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                log(logging.WARNING, "No Hugging Face token found in config.yaml or HF_TOKEN environment variable.")
                log(logging.WARNING, "You may encounter 401 authentication errors when accessing Hugging Face models.")
                log(logging.WARNING, "Please update config.yaml with your token from https://huggingface.co/settings/tokens")
        
        # Configure diarizer
        diarization_config = {
            'debug': args.debug or args.debug_files_only,
            'debug_dir': debug_dir if (args.debug or args.debug_files_only) else None,
            'min_speakers': args.min_speakers,
            'max_speakers': args.max_speakers,
            'clustering_threshold': args.clustering_threshold,
            'use_gpu': torch.cuda.is_available(),
            'huggingface_token': hf_token,
            'batch_size': 32
        }
        
        # Override speaker count if specified
        # 2025-04-24 -JS
        if args.speaker_count:
            log(logging.INFO, f"Using specified speaker count: {args.speaker_count}")
            diarization_config['min_speakers'] = args.speaker_count
            diarization_config['max_speakers'] = args.speaker_count
        
        # Import diarizer here to ensure the mock in tests works correctly
        from src.audio_processing.diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(diarization_config)
        
        # Run diarization
        # 2025-04-24 -JS
        # If continuing from diarization or srt, look for existing segments files
        existing_segments = None
        if args.continue_from in ['diarization', 'srt']:
            # For SRT generation, we need to have processed audio file
            if args.continue_from == 'srt' and not os.path.exists(os.path.join(run_dir, f"{input_basename}_processed.wav")):
                # Look for processed WAV files
                processed_wavs = glob.glob(os.path.join(run_dir, "*_processed.wav"))
                if processed_wavs:
                    # Use the first processed WAV file as input
                    input_file = processed_wavs[0]
                    input_basename = os.path.splitext(os.path.basename(input_file))[0].replace("_processed", "")
                    log(logging.INFO, f"Using processed WAV file: {input_file}")
            # Find all segments files in the folder
            segments_files = glob.glob(os.path.join(run_dir, "*.segments"))
            
            # If speaker count is specified, filter for that count
            if args.speaker_count and args.continue_from == 'diarization':
                speaker_segments = [f for f in segments_files if f"{args.speaker_count}speakers" in os.path.basename(f)]
                if speaker_segments:
                    segments_files = speaker_segments
            
            if len(segments_files) == 1:
                existing_segments = segments_files[0]
                log(logging.INFO, f"Found segments file: {os.path.basename(existing_segments)}")
            elif len(segments_files) > 1:
                log(logging.INFO, f"Found {len(segments_files)} segments files. Please select one:")
                for i, file in enumerate(segments_files):
                    print(f"  [{i+1}] {os.path.basename(file)}")
                
                while True:
                    try:
                        choice = input("Enter the number of the segments file to use: ")
                        idx = int(choice) - 1
                        if 0 <= idx < len(segments_files):
                            existing_segments = segments_files[idx]
                            log(logging.INFO, f"Selected segments file: {os.path.basename(existing_segments)}")
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(segments_files)}")
                    except ValueError:
                        print("Please enter a valid number")
        
        if existing_segments:
            # Load existing segments file
            diarization_segments = diarizer.load_segments(existing_segments)
            log(logging.INFO, f"Loaded existing diarization segments from {existing_segments}")
        elif args.continue_from == 'srt':
            # If continuing from SRT and no segments file found, we can't continue
            log(logging.ERROR, "No segments file found. Cannot continue with SRT generation.")
            return False
        else:
            # Run diarization normally
            diarization_segments = diarizer.diarize(input_file, run_dir)
            if not diarization_segments:
                log(logging.ERROR, "Diarization failed")
                return False
        
        log(logging.INFO, "Speaker diarization completed successfully")
    
    # Generate SRT file unless explicitly skipped
    # 2025-04-24 -JS
    if not args.skip_srt and (diarization_segments or args.continue_from == 'srt'):
        # If continuing from SRT and we don't have diarization_segments yet, load them
        if args.continue_from == 'srt' and not diarization_segments:
            # Find all segments files in the folder
            segments_files = glob.glob(os.path.join(run_dir, "*.segments"))
            
            if len(segments_files) == 1:
                existing_segments = segments_files[0]
                log(logging.INFO, f"Found segments file: {os.path.basename(existing_segments)}")
            elif len(segments_files) > 1:
                log(logging.INFO, f"Found {len(segments_files)} segments files. Please select one:")
                for i, file in enumerate(segments_files):
                    print(f"  [{i+1}] {os.path.basename(file)}")
                
                while True:
                    try:
                        choice = input("Enter the number of the segments file to use: ")
                        idx = int(choice) - 1
                        if 0 <= idx < len(segments_files):
                            existing_segments = segments_files[idx]
                            log(logging.INFO, f"Selected segments file: {os.path.basename(existing_segments)}")
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(segments_files)}")
                    except ValueError:
                        print("Please enter a valid number")
            else:
                log(logging.ERROR, "No segments file found. Cannot continue with SRT generation.")
                return False
            
            # Import diarizer here to ensure the mock in tests works correctly
            from src.audio_processing.diarization import SpeakerDiarizer
            diarizer = SpeakerDiarizer({})
            
            # Load segments file
            diarization_segments = diarizer.load_segments(existing_segments)
            log(logging.INFO, f"Loaded diarization segments from {existing_segments}")
            
            # Transcribe the segments and generate SRT file directly
            # 2025-04-24 -JS
            if args.skip_transcription:
                log(logging.INFO, "Skipping transcription as requested")
            elif input_file and os.path.exists(input_file):
                log(logging.INFO, "Transcribing audio segments and generating SRT file...")
                try:
                    # Import the transcriber
                    from src.audio_processing.transcriber import WhisperTranscriber
                    
                    # Create transcriber with Swedish language and KBLab model
                    # 2025-04-24 -JS
                    transcriber = WhisperTranscriber({
                        'language': 'sv',  # Swedish language
                        'model_name': 'KBLab/kb-whisper-large',  # Use KBLab model optimized for Swedish
                        'pre_padding': args.srt_pre,  # Add padding before segment
                        'post_padding': args.srt_post,  # Add padding after segment
                        'confidence_threshold': args.confidence_threshold,  # Minimum confidence threshold
                        'min_segment_duration': args.srt_min_duration,  # Minimum segment duration
                        'include_speaker': not args.srt_no_speaker,  # Whether to include speaker labels
                        'remove_empty_segments': args.srt_remove_empty,  # Whether to remove empty segments
                        'empty_placeholder': args.srt_empty_placeholder  # Placeholder text for empty segments
                    })
                    
                    # Generate SRT file path
                    srt_file = os.path.join(run_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.srt")
                    
                    # Limit segments if max_segments is specified
                    if args.max_segments and args.max_segments > 0:
                        log(logging.INFO, f"Limiting transcription to {args.max_segments} segments")
                        segments_to_transcribe = diarization_segments[:args.max_segments]
                    else:
                        segments_to_transcribe = diarization_segments
                    
                    # Transcribe segments and write directly to SRT file
                    # 2025-04-24 -JS - Added segment count tracking
                    result, processed_count, filtered_count = transcriber.transcribe_segments_to_srt(input_file, segments_to_transcribe, srt_file)
                    
                    if result:
                        log(logging.INFO, f"SRT file generated successfully: {srt_file}")
                        log(logging.INFO, f"Processed {processed_count} segments, filtered {filtered_count} segments")
                        log(logging.INFO, "SRT generation completed")
                        return True
                    else:
                        log(logging.ERROR, "SRT generation failed")
                except Exception as e:
                    log(logging.ERROR, f"Error during transcription: {str(e)}")
                    log(logging.WARNING, "Falling back to SRT generation without transcription")
            else:
                log(logging.WARNING, "No audio file available for transcription. SRT will be generated without text.")
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
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                log(logging.DEBUG, "FFmpeg command is available")
                
                # Now check for specific libraries with version numbers
                ffmpeg_found = False
                
                # Check for libavutil with different version numbers
                avutil_found = False
                for version in ["58", "57", "56", "55", ""]:
                    try:
                        lib_name = f"libavutil.so.{version}" if version else "libavutil.so"
                        ctypes.CDLL(lib_name)
                        avutil_found = True
                        log(logging.DEBUG, f"Found FFmpeg library: {lib_name}")
                        break
                    except OSError:
                        log(logging.DEBUG, f"Could not find FFmpeg library: {lib_name}")
                
                # Check for libavcodec with different version numbers
                avcodec_found = False
                for version in ["60", "59", "58", ""]:
                    try:
                        lib_name = f"libavcodec.so.{version}" if version else "libavcodec.so"
                        ctypes.CDLL(lib_name)
                        avcodec_found = True
                        log(logging.DEBUG, f"Found FFmpeg library: {lib_name}")
                        break
                    except OSError:
                        log(logging.DEBUG, f"Could not find FFmpeg library: {lib_name}")
                
                # Check for libavformat with different version numbers
                avformat_found = False
                for version in ["60", "59", "58", ""]:
                    try:
                        lib_name = f"libavformat.so.{version}" if version else "libavformat.so"
                        ctypes.CDLL(lib_name)
                        avformat_found = True
                        log(logging.DEBUG, f"Found FFmpeg library: {lib_name}")
                        break
                    except OSError:
                        log(logging.DEBUG, f"Could not find FFmpeg library: {lib_name}")
                
                ffmpeg_found = avutil_found and avcodec_found and avformat_found
                
                if ffmpeg_found:
                    log(logging.DEBUG, "All required FFmpeg libraries found")
                else:
                    missing_libs = []
                    if not avutil_found: missing_libs.append("libavutil")
                    if not avcodec_found: missing_libs.append("libavcodec")
                    if not avformat_found: missing_libs.append("libavformat")
                    
                    warning_msg = f"FFmpeg command is available but libraries not found: {', '.join(missing_libs)}"
                    log(logging.WARNING, warning_msg)
                    print(f"\033[93mWARNING: {warning_msg}\033[0m")
                    print("\033[93m       This may cause issues with audio processing\033[0m")
                    print("\033[93m       Install the missing FFmpeg development libraries:\033[0m")
                    print("\033[93m       sudo apt-get install libavutil-dev libavcodec-dev libavformat-dev\033[0m")
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

def list_available_steps():
    """
    Display all available processing steps that can be skipped.
    
    2025-04-24 -JS
    """
    print("\033[1mAvailable processing steps that can be skipped:\033[0m")
    print("\nAudio preprocessing steps:")
    print("  - highpass             : Skip high-pass filter (preserves low frequencies)")
    print("  - highpass_compensation: Skip volume compensation after high-pass filter")
    print("  - lowpass              : Skip low-pass filter (preserves high frequencies)")
    print("  - lowpass_compensation : Skip volume compensation after low-pass filter")
    print("  - compression          : Skip dynamic range compression")
    print("  - normalize            : Skip audio normalization")
    print("  - volume               : Skip final volume adjustment")
    
    print("\nUsage examples:")
    print("  --skip-steps highpass,lowpass          : Skip both filters but keep other processing")
    print("  --skip-steps compression,normalize     : Skip compression and normalization")
    print("  --use-vocals-directly                 : Skip all post-processing steps")
    print("\nNote: Steps are applied in the order listed above.")

def main():
    """
    Main entry point for the audio toolkit.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Handle --list-steps option
    if args.list_steps:
        list_available_steps()
        return True
    
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
