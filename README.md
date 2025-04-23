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

- **Speaker Diarization**: Separate audio by speaker using advanced models
- **Voice Activity Detection**: Identify speech regions and filter out background noise
- **Multi-stage Processing**: Combine specialized models for better results
- **GPU Optimization**: Utilize GPU acceleration for faster processing
- **Swedish Dialect Optimization**: Special parameters tuned for similar Swedish dialects

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
