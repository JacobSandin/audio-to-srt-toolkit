# Changelog

All notable changes to this project will be documented in this file.

## [0.0.021] - 2025-04-23

### Changed
- Enhanced Demucs progress reporting to support the latest htdemucs model
- Added support for tracking progress across multiple models in model bags
- Improved progress milestone reporting with 10% increments (0%, 10%, 20%, etc.)
- Fixed issue where Demucs progress was not being displayed with newer models

## [0.0.020] - 2025-04-23

### Changed
- Added progress reporting to audio compression process with 10% milestone updates
- Implemented custom compression algorithm with detailed progress logging
- Fixed issue where compression step had no progress feedback during long processing

## [0.0.019] - 2025-04-23

### Changed
- Optimized highpass filter cutoff frequency to 3750Hz based on extensive testing
- Updated README.md with new findings about optimal filter settings for Swedish dialects
- Testing showed best results between 3500-4000Hz for dialect clarity and distinction

## [0.0.018] - 2025-04-23

### Changed
- Improved file naming convention with consistent patterns and single timestamps
- Created test_highpass_cutoffs.py script for comparing different filter settings
- Fixed multiple timestamp issue in debug and output files
- Organized output files with timestamped directories for better comparison

## [0.0.017] - 2025-04-23

### Changed
- Enhanced README.md with detailed audio processing pipeline documentation
- Added comprehensive flow diagram showing all processing steps
- Expanded descriptions of each processing stage with parameters and purpose
- Added information about highpass filter cutoff options (150-450Hz)

## [0.0.016] - 2025-04-23

### Changed
- Improved progress reporting in Demucs vocal separation to reduce log clutter
- Enhanced logging with cleaner milestone reporting (0%, 25%, 50%, 75%, 100%)
- Added better categorization of log messages by importance level
- Improved error and warning detection in process output

## [0.0.015] - 2025-04-23

### Changed
- Improved .windsurfrules structure and clarity
- Enhanced formatting with clear section dividers
- Added explicit separation between project-specific and global rules
- Improved startup behavior description with numbered steps
- Added implementation notes for better clarity
- Updated version in all files to match CHANGELOG.md

## [0.0.014] - 2025-04-23

### Changed
- Restructured CHANGELOG.md to track each version increment (0.0.001 to 0.0.014)
- Added __version__ variable to package __init__.py files
- Created top-level __init__.py with version information
- Implemented semantic versioning (major.minor.patch) format
- Each commit now has its own version number in the changelog

### Changed
- Updated version numbering system to properly track development progress
- Added __version__ variable to package __init__.py files
- Created top-level __init__.py with version information

## [0.0.013] - 2025-04-23

### Added
- Enhanced progress reporting for audio processing
- Added detailed real-time feedback for all processing steps including Demucs vocal separation
- Added percentage progress reporting for lengthy operations
- Improved logging with time estimates and file information

## [0.0.012] - 2025-04-23

### Changed
- Improved intermediate file handling: now using temporary files when not in debug mode
- Moved large intermediate files (highquality.wav, vocals.wav) to debug directory with timestamps
- Ensured all output files use WAV format for high quality
- Implemented automatic cleanup of temporary files

## [0.0.011] - 2025-04-23

### Added
- Added .windsurfrules with project configuration for audio processing models
- Specified preferred models for Swedish dialect processing

## [0.0.010] - 2025-04-23

### Added
- Enhanced model loading with multiple model support
- Using tensorlake/speaker-diarization-3.1 as primary model for Swedish dialects
- Implemented fallback mechanism for model loading with multiple options
- Improved progress reporting during diarization and SRT generation

## [0.0.009] - 2025-04-23

### Added
- SRT subtitle file generation from diarization results
- Command-line arguments for customizing SRT output (--generate-srt, --include-timestamps, --speaker-format, --max-gap, --max-duration)
- Segment merging functionality to combine consecutive segments from the same speaker

## [0.0.008] - 2025-04-23

### Changed
- Updated debug output files to use WAV format instead of MP3
- Added warning filters to suppress all dependency-related warnings for clean test output
- Added pytest configuration file (conftest.py) to ensure consistent warning suppression across all tests

## [0.0.007] - 2025-04-23

### Added
- Speaker diarization feature optimized for Swedish dialects
- Command-line arguments for diarization parameters (--diarize, --min-speakers, --max-speakers, --clustering-threshold)
- Optimized audio quality throughout the processing pipeline

## [0.0.006] - 2025-04-23

### Changed
- Reduced default volume gain from 6.0 dB to 3.0 dB for better audio quality

## [0.0.005] - 2025-04-23

### Added
- High-quality WAV conversion as first preprocessing step
- Configurable bit depth (16/24/32-bit) and sample rate (44.1/48/96kHz)

## [0.0.004] - 2025-04-23

### Added
- Created test-driven development structure with tests/ and src/ directories
- Added requirements.txt with necessary dependencies

## [0.0.003] - 2025-04-23

### Added
- Added debug mode with intermediate file output for each processing step
- Enhanced logging for better troubleshooting

## [0.0.002] - 2025-04-23

### Added
- Initial project setup with old_tests directory containing all scripts
- Speaker diarization scripts with pyannote/speaker-diarization-3.1 model
- Advanced diarization script with multi-stage processing for Swedish dialects

## [0.0.001] - 2025-04-23

### Added
- Initial commit
- Set up Swedish dialect audio processing toolkit with scripts for diarization and transcription


