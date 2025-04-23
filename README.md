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
- **GPU Optimization**: Utilize GPU acceleration for faster processing
- **Swedish Dialect Optimization**: Special parameters tuned for similar Swedish dialects
- **Debug Mode**: Save intermediate files at each processing step for troubleshooting

## Requirements

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

# With audio processing options
./audio_toolkit.py --input-audio your_audio_file.mp3 --highpass 200 --lowpass 7000 --volume-gain 4

# With debug mode (saves intermediate files for each processing step)
./audio_toolkit.py --input-audio your_audio_file.mp3 --debug
```

The toolkit performs the following preprocessing steps:
1. Vocal separation using demucs
2. Audio normalization
3. High-pass filtering to emphasize dialect differences
4. Compression to balance audio levels
5. Volume adjustment to ensure voices are clearly audible

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
