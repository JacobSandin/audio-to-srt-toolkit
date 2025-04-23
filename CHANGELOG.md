# Changelog

All notable changes to this project will be documented in this file.

## [0.0.002] - 2025-04-23

### Added
- Initial project setup with old_tests directory containing all scripts
- Speaker diarization scripts with pyannote/speaker-diarization-3.1 model
- Advanced diarization script with multi-stage processing for Swedish dialects
- Voice Activity Detection using pyannote/voice-activity-detection
- GPU optimization with TF32 acceleration and memory management
- Support for distinguishing between speakers with similar dialects
- Audio preprocessing capabilities to enhance dialect differences
- Added .gitignore to exclude audio files and personal data
- Created comprehensive README.md with usage instructions
- Implemented unified audio_toolkit.py command with modular architecture
- Created test-driven development structure with tests/ and src/ directories
- Added audio preprocessing pipeline with demucs, normalization, filtering, and compression
- Added debug mode with intermediate file output for each processing step

## [0.0.000] - Former Unreleased

### Added
- High-quality WAV conversion as first preprocessing step
- Configurable bit depth (16/24/32-bit) and sample rate (44.1/48/96kHz)
- Speaker diarization feature optimized for Swedish dialects
- Command-line arguments for diarization parameters (--diarize, --min-speakers, --max-speakers, --clustering-threshold)
- SRT subtitle file generation from diarization results
- Command-line arguments for SRT generation (--generate-srt, --include-timestamps, --speaker-format, --max-gap, --max-duration)
- Segment merging functionality to combine consecutive segments from the same speaker
- Enhanced model loading with multiple model support (tensorlake/speaker-diarization-3.1 as primary model)
- Improved progress reporting during diarization and SRT generation
- Added detailed real-time feedback for all processing steps including Demucs vocal separation
- Added percentage progress reporting for lengthy operations

### Changed
- Changed output format from MP3 to WAV to maintain high audio quality throughout the processing pipeline
- Updated debug output files to use WAV format instead of MP3
- Updated matplotlib API usage to avoid deprecation warnings
- Updated torchaudio API usage to avoid deprecation warnings
- Improved intermediate file handling: now using temporary files when not in debug mode
- Moved large intermediate files (highquality.wav, vocals.wav) to debug directory with timestamps when in debug mode
- Updated code to use newer APIs for torchaudio and matplotlib to reduce deprecation warnings
- Added warning filters to suppress all dependency-related warnings for clean test output
- Added pytest configuration file (conftest.py) to ensure consistent warning suppression across all tests

## [0.0.000] - Former Unreleased

### Added
- Experimented with different diarization approaches
- Tested various preprocessing methods for audio enhancement
- Explored manual diarization for difficult cases
- Implemented progress tracking during diarization
