# Changelog

All notable changes to this project will be documented in this file.

## [0.1.000] - 2025-04-23

### Added
- Initial project setup with old_tests directory containing all scripts
- Speaker diarization scripts with pyannote/speaker-diarization-3.1 model
- Advanced diarization script with multi-stage processing for Swedish dialects
- Voice Activity Detection using pyannote/voice-activity-detection
- GPU optimization with TF32 acceleration and memory management
- Support for distinguishing between similar Swedish dialects
- Audio preprocessing capabilities to enhance dialect differences
- Added .gitignore to exclude audio files and personal data
- Created comprehensive README.md with usage instructions

### Changed
- Organized all scripts in the old_tests directory
- Updated command syntax to use "if test $status -eq 0" instead of "if [ $? -eq 0 ]"

## [0.0.000] - Former Unreleased

### Added
- Experimented with different diarization approaches
- Tested various preprocessing methods for audio enhancement
- Explored manual diarization for difficult cases
- Implemented progress tracking during diarization
