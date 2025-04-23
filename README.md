# Swedish Dialect Audio Processing Toolkit

A toolkit for processing audio recordings of Swedish dialects, with a focus on speaker diarization and transcription.

## Overview

This project provides tools for processing audio recordings of Swedish dialects, particularly focusing on distinguishing between speakers with similar dialects and minimizing background noise interference (such as motorcycle noise).

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
git clone https://github.com/yourusername/swedish-dialect-audio-toolkit.git
cd swedish-dialect-audio-toolkit

# Install dependencies
pip install -r requirements.txt

# Set up Hugging Face token (required for model access)
export HF_TOKEN=your_huggingface_token
```

## Usage

### Speaker Diarization

```bash
python diarization.py your_audio_file.mp3
```

This will create a `.segments` file with speaker timestamps that can be used with transcription tools.

### Advanced Diarization (for similar dialects)

```bash
python advanced_diarization.py your_audio_file.mp3
```

This uses a multi-stage approach with separate models for VAD and diarization, and tries multiple speaker counts to find the optimal configuration.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) for the speaker diarization models
- [Hugging Face](https://huggingface.co/) for model hosting
