# Swedish Dialect Audio Processing Toolkit

A toolkit for processing audio recordings of Swedish dialects, with a focus on speaker diarization and transcription.

## Overview

This project provides tools for processing audio recordings of Swedish dialects, particularly focusing on distinguishing between speakers with similar dialects and minimizing background noise interference (such as motorcycle noise).

## Project Structure

The project contains the following key components:

- **old_tests/**: Contains all the Python scripts for audio processing
  - **diarization.py**: Main speaker diarization script
  - **advanced_diarization.py**: Multi-stage diarization for similar dialects
  - **fast-whisper-minimal-segments.py**: Transcription with speaker labels
  - **preprocess_audio.py**: Audio preprocessing utilities
  - **preprocess_with_demucs.py**: Background noise removal
  - **split_audio.py**: Audio splitting utilities

## Features

- **High-Quality Audio Processing**: Convert to high-bit-depth WAV format for better dialect separation
- **Speaker Diarization**: Separate audio by speaker using advanced models
- **Voice Activity Detection**: Identify speech regions and filter out background noise
- **Multi-stage Processing**: Combine specialized models for better results
- **GPU Optimization**: Utilize GPU acceleration for faster processing with dynamic memory management
- **Swedish Dialect Optimization**: Special parameters tuned for similar Swedish dialects
- **Debug Mode**: Save intermediate files at each processing step for troubleshooting

## Requirements

### FFmpeg Libraries
This toolkit relies on FFmpeg for audio processing. You must have the correct FFmpeg libraries installed on your system:

```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev libswresample-dev

# For Fedora/RHEL/CentOS
sudo dnf install ffmpeg ffmpeg-devel

# For macOS (using Homebrew)
brew install ffmpeg
```

If you encounter errors like `libavutil.so.58: cannot open shared object file`, it means the required FFmpeg libraries are missing.

### Python Dependencies
Install Python dependencies using:

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Hugging Face account with access to pyannote models

## Installation

```bash
# Clone the repository
git clone https://github.com/JacobSandin/audio-to-srt-toolkit.git
cd audio-to-srt-toolkit

# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token (required for model access)
export HF_TOKEN=your_huggingface_token
```

## Troubleshooting

If you encounter issues with FFmpeg dependencies, diarization, or performance, please refer to the [FAQ.md](FAQ.md) file for detailed solutions to common problems.

The FAQ covers:
- FFmpeg library dependency issues and solutions
- Diarization troubleshooting
- Optimal filter settings for Swedish dialects
- Performance optimization tips

## Usage

### Audio Toolkit Command

The `audio_toolkit.py` script provides a unified command-line interface for all audio processing tasks:

```bash
# Basic usage
./audio_toolkit.py --input-audio your_audio_file.mp3

# With custom output directory
./audio_toolkit.py --input-audio your_audio_file.mp3 --output-dir /path/to/output

# With high-quality WAV conversion options (for better dialect separation)
./audio_toolkit.py --input-audio your_audio_file.mp3 --bit-depth 24 --sample-rate 48000

# With audio processing options (default highpass is now 3750Hz, optimal for Swedish dialects)
./audio_toolkit.py --input-audio your_audio_file.mp3 --lowpass 7000 --volume-gain 4

# With custom highpass filter (lower values retain more bass, higher values isolate speech better)
./audio_toolkit.py --input-audio your_audio_file.mp3 --highpass 4000 --lowpass 7000

# With speaker diarization optimized for Swedish dialects
./audio_toolkit.py --input-audio your_audio_file.mp3 --diarize --min-speakers 2 --max-speakers 4 --clustering-threshold 0.65

# Generate SRT subtitle file from diarization results
./audio_toolkit.py --input-audio your_audio_file.mp3 --diarize --generate-srt

# Generate SRT with custom speaker format and timestamps
./audio_toolkit.py --input-audio your_audio_file.mp3 --diarize --generate-srt --speaker-format "Person {speaker_id}:" --include-timestamps

# Merge consecutive segments from the same speaker with custom gap and duration limits
./audio_toolkit.py --input-audio your_audio_file.mp3 --diarize --generate-srt --max-gap 1.5 --max-duration 15.0

# With debug mode (saves intermediate files for each processing step)
./audio_toolkit.py --input-audio your_audio_file.mp3 --debug

# List all available processing steps that can be skipped
./audio_toolkit.py --list-steps

# Skip specific processing steps (e.g., highpass and lowpass filters)
./audio_toolkit.py --input-audio your_audio_file.mp3 --skip-steps highpass,lowpass

# Use vocals file directly for transcription (skip post-processing steps)
./audio_toolkit.py --input-audio your_audio_file.mp3 --use-vocals-directly

# Combined preprocessing, diarization, and SRT generation with debug output
./audio_toolkit.py --input-audio your_audio_file.mp3 --diarize --generate-srt --debug

# Continue processing from diarization step with a specific speaker count (input file is auto-detected)
./audio_toolkit.py --continue-from diarization --continue-folder /path/to/output/20250424_123456_filename --speaker-count 3

# Continue processing from SRT generation step
./audio_toolkit.py --continue-from srt --continue-folder /path/to/output/20250424_123456_filename

# Continue processing with debug files only (useful for troubleshooting)
./audio_toolkit.py --continue-from diarization --continue-folder /path/to/output/20250424_123456_filename --debug-files-only --speaker-count 3
```

## GPU Acceleration and Optimization

The toolkit now features advanced GPU optimization for both diarization models and Demucs vocal separation:

### Demucs GPU Optimization

- **Dynamic Memory Management**: Automatically adjusts processing parameters based on available GPU memory
- **Tiered Memory Settings**: Optimized configurations for different GPU capacities (16GB+, 8-16GB, 4-8GB, <4GB)
- **Segment Overlap**: Implements variable overlap parameters (0.1-0.25) for improved quality without memory issues
- **Float32 Precision**: Uses optimized floating-point precision for better GPU performance
- **Parallel Processing**: Configures optimal thread count for CPU/GPU parallelization

To test GPU performance with Demucs:

```bash
# Test GPU acceleration with a 60-second sample
python test_demucs_gpu.py --duration 60

# Process a specific audio file with GPU monitoring
python test_demucs_gpu.py --input your_audio_file.mp3 --duration 120
```

## Handling Interruptions and Crashes

The toolkit now supports continuing from a previous run if the process was interrupted or crashed. This is particularly useful for long audio files or when trying different speaker counts for diarization.

### Continuing After a Crash

If the diarization process crashes (e.g., due to memory issues or segmentation faults), you can continue from where it left off:

```bash
# Continue from diarization with a different speaker count
./audio_toolkit.py --continue-from diarization --continue-folder out/20250424_123456_filename --speaker-count 3
```

### Trying Different Speaker Counts

If you're not sure how many speakers are in the audio, you can try different counts sequentially:

```bash
# First try with 2 speakers
./audio_toolkit.py --input-audio your_audio_file.mp3 --debug-files-only

# If that doesn't work well, try with 3 speakers
./audio_toolkit.py --continue-from diarization --continue-folder out/20250424_123456_filename --speaker-count 3

# If needed, try with 4 speakers
./audio_toolkit.py --continue-from diarization --continue-folder out/20250424_123456_filename --speaker-count 4
```

### Skipping to SRT Generation

If diarization completed but you want to regenerate the SRT file with different settings:

```bash
# Regenerate SRT with different settings
./audio_toolkit.py --continue-from srt --continue-folder out/20250424_123456_filename --include-timestamps --max-gap 2.0
```

### Interactive File Selection

When continuing from a previous run with multiple segments files or audio files, the toolkit will now interactively prompt you to select which file to use:

```bash
# Continue from a folder with multiple segments files
./audio_toolkit.py --continue-from srt --continue-folder out/20250424_123456_filename

# You'll see a prompt like this:
Found 3 segments files. Please select one:
  [1] audio_processed.2speakers.segments
  [2] audio_processed.3speakers.segments
  [3] audio_processed.4speakers.segments
Enter the number of the segments file to use: 
```

This is particularly useful when you've tried diarization with different speaker counts and want to choose the best one for SRT generation.

## Audio Processing Pipeline

The toolkit implements a comprehensive audio processing pipeline optimized for Swedish dialect analysis:

```
Input Audio → High-Quality WAV → Vocal Separation → Filtering → Normalization → Compression → Volume Adjustment → Diarization → SRT Generation
```

### Flexible Processing Options

The toolkit provides several options to customize the processing pipeline based on your audio quality needs:

- **Selective Step Skipping**: Use `--skip-steps` to skip specific processing steps (e.g., `--skip-steps highpass,lowpass`) when they might degrade audio quality
- **Direct Vocals Usage**: Use `--use-vocals-directly` to skip all post-processing steps and use the vocals file directly for transcription
- **Step Listing**: Use `--list-steps` to see all available processing steps that can be skipped

These options are particularly useful when working with high-quality audio recordings where certain processing steps might remove important dialect characteristics.

### Detailed Processing Steps

1. **High-Quality WAV Conversion**
   - Converts input audio to WAV format with configurable bit depth (16/24/32-bit) and sample rate (44.1/48/96kHz)
   - Higher bit depth and sample rate preserve subtle dialect characteristics
   - Default: 24-bit, 48kHz for optimal quality-to-size ratio

2. **Vocal Separation (Demucs)**
   - Uses Demucs deep learning model to separate vocals from background elements
   - Particularly effective at removing environmental noise (traffic, wind, etc.)
   - Preserves vocal characteristics critical for dialect analysis
   - Progress reporting at 0%, 25%, 50%, 75%, and 100% milestones

3. **High-Pass Filtering**
   - Removes low-frequency noise below the cutoff frequency
   - Configurable cutoff (default: 3750Hz, optimized for Swedish dialect isolation)
   - Testing showed optimal results between 3500-4000Hz for Swedish speech
   - Higher cutoffs (3000-4000Hz) significantly improve speech clarity and dialect distinction
   - Especially effective for isolating subtle dialect characteristics from background noise

4. **Low-Pass Filtering**
   - Removes high-frequency noise above the cutoff frequency
   - Configurable cutoff (default: 8000Hz)
   - Helps reduce hissing and sibilance while preserving speech clarity

5. **Audio Normalization**
   - Balances audio levels across the entire recording
   - Makes quiet sections more audible without distorting louder sections
   - Ensures consistent volume levels for better diarization accuracy

6. **Dynamic Range Compression**
   - Reduces the difference between loudest and quietest parts
   - Makes softer speech more audible (important for dialect analysis)
   - Configurable threshold and ratio
   - Default: -10dB threshold with 2:1 ratio

7. **Volume Adjustment**
   - Final gain adjustment to optimize listening level
   - Configurable gain in dB (default: +3dB)

8. **Speaker Diarization**
   - Uses advanced ML models to identify and separate different speakers
   - Optimized for Swedish dialects with similar speech patterns
   - Multi-stage approach with specialized models for Voice Activity Detection (VAD) and diarization
   - Configurable speaker count parameters (min/max speakers)
   - Clustering threshold optimized for distinguishing similar dialects

9. **SRT Generation**
   - Creates subtitle files from diarization results
   - Configurable speaker labeling and timestamp formats
   - Segment merging to combine consecutive utterances from the same speaker
   - Customizable gap and duration parameters

All processing maintains the high-quality WAV format throughout the pipeline to preserve audio quality for dialect analysis. Debug mode saves intermediate files at each step for analysis and troubleshooting.

### File Organization and Naming Convention

- **Output Directory Structure**: The toolkit creates timestamped directories for each processing run
  ```
  out/
  ├── 20250424_195130_Cardo recording 1/  # Timestamped directory for each run
  │   ├── debug/                           # Debug files for each processing step
  │   │   ├── 01_wav_conversion_Cardo recording 1.wav
  │   │   ├── 02_vocals_Cardo recording 1.wav
  │   │   └── ...
  │   ├── Cardo recording 1_processed.wav  # Processed audio file (no timestamp)
  │   ├── Cardo recording 1.segments       # Diarization segments file
  │   └── Cardo recording 1.srt            # Generated SRT file
  └── ...
  ```

- **Naming Convention**: 
  - Directories have timestamps (e.g., `20250424_195130_Cardo recording 1/`)
  - Files inside timestamped directories don't have timestamps in their names
  - Debug files are prefixed with step numbers (e.g., `01_wav_conversion_`)
  - This organization keeps the file structure clean and logical

### Processing Flow Diagram

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌───────────┐
│  Input      │────▶│ High-Quality │────▶│  Vocal      │────▶│ High-Pass │
│  Audio File │     │  WAV         │     │  Separation │     │ Filter    │
└─────────────┘     └──────────────┘     └─────────────┘     └───────────┘
        │                                                           ▼
        │                                                    ┌───────────┐
        │                                                    │ Low-Pass  │
        │                                                    │ Filter    │
        │                                                    └───────────┘
        │                                                           ▼
        │                                                    ┌───────────┐
        │                                                    │ Normalize │
        │                                                    │ Audio     │
        │                                                    └───────────┘
        │                                                           ▼
        │                                                    ┌───────────┐
        │                                                    │ Compress  │
        │                                                    │ Dynamic   │
        │                                                    │ Range     │
        │                                                    └───────────┘
        │                                                           ▼
        │                                                    ┌───────────┐
        │                                                    │ Adjust    │
        │                                                    │ Volume    │
        │                                                    └───────────┘
        │                                                           ▼
        ▼                                                    ┌───────────┐
┌─────────────┐     ┌──────────────┐     ┌─────────────┐    │ Processed │
│  Speaker    │◀────│ Voice        │◀────│ Diarization │◀───│ Audio     │
│  Segments   │     │ Activity     │     │ Model       │    │ File      │
└─────────────┘     │ Detection    │     └─────────────┘    └───────────┘
        │            └──────────────┘
        ▼
┌─────────────┐
│  SRT        │
│  Subtitles  │
└─────────────┘
```

For diarization, the toolkit uses the following approach:

1. **Voice Activity Detection (VAD)**: Identifies segments containing speech
2. **Speaker Diarization**: Identifies different speakers in the audio
3. **Optimization for Swedish Dialects**: Uses higher speaker counts and specialized parameters

## SRT Generation

The toolkit can generate SRT subtitle files from diarization results with the following features:

1. **Speaker Labeling**: Each segment is labeled with a speaker identifier (e.g., "SPEAKER_01")
2. **Timestamp Formatting**: Converts start and end times to SRT timestamp format (HH:MM:SS,mmm)
3. **Segment Merging**: Combines consecutive segments from the same speaker within a configurable gap
4. **Custom Formatting**: Allows customization of speaker labels and optional inclusion of timestamps

### SRT Generation Parameters

- `--generate-srt`: Enable SRT generation (requires diarization to be enabled)
- `--include-timestamps`: Include timestamps in the subtitle text (default: False)
- `--speaker-format`: Format string for speaker labels (default: "{speaker}:")
- `--max-gap`: Maximum gap in seconds between segments to merge (default: 1.0)
- `--max-duration`: Maximum duration in seconds for merged segments (default: 10.0)
- `--srt-pre`: Seconds to add before each segment for transcription (default: 0.1)
- `--srt-post`: Seconds to add after each segment for transcription (default: 0.1)
- `--srt-min-duration`: Minimum segment duration in seconds to include in SRT (default: 0.3)
- `--srt-no-speaker`: Remove speaker labels from SRT output
- `--confidence-threshold`: Minimum confidence threshold for transcriptions (0.0-1.0, default: 0.5)

### Debug Mode

When running with the `--debug` flag, the toolkit will save intermediate audio files for each processing step in a `debug` subdirectory of your output directory. Each file includes:

- The original filename
- The processing step name
- A timestamp

For example:
```
debug/recording.vocals.20250423-140815.mp3
debug/recording.highpass.20250423-140815.mp3
debug/recording.lowpass.20250423-140815.mp3
debug/recording.compression.20250423-140815.mp3
debug/recording.normalize.20250423-140815.mp3
debug/recording.volume.20250423-140815.mp3
```

This is useful for diagnosing issues with specific processing steps or fine-tuning parameters.

### Speaker Diarization

```bash
python old_tests/diarization.py your_audio_file.mp3
```

This will create a `.segments` file with speaker timestamps that can be used with transcription tools.

### Advanced Diarization (for similar dialects)

```bash
python old_tests/advanced_diarization.py your_audio_file.mp3
```

This uses a multi-stage approach with separate models for VAD and diarization, and tries multiple speaker counts to find the optimal configuration.

### Transcription with Speaker Labels

```bash
python old_tests/fast-whisper-minimal-segments.py your_audio_file.mp3 your_audio_file.segments
```

## Testing

When testing command execution status, use the following pattern:

```bash
command_to_run
if test $status -eq 0
then
    echo "Command succeeded"
else
    echo "Command failed"
fi
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) for the speaker diarization models
- [Hugging Face](https://huggingface.co/) for model hosting
