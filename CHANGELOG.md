# Changelog

All notable changes to this project will be documented in this file.

## [0.0.061] - 2025-04-24

### Changed
- Changed padding application order: now applies --srt-pre and --srt-post padding after minimum duration check
- Improved filtering efficiency by checking minimum duration before transcription
- Separated --srt-min-duration and --srt-remove-empty checks for better control

## [0.0.060] - 2025-04-24

### Added
- Added `--srt-remove-empty` option to explicitly control removal of empty segments
- Added `--srt-empty-placeholder` option with default value "[UNRECOGNIZABLE]" for empty segments

### Fixed
- Fixed issue where empty segments weren't included in SRT output even with --srt-min-duration 0
- Made empty segment filtering respect the --srt-min-duration parameter
- Ensured empty segments are properly removed when --srt-remove-empty is used or --srt-min-duration > 0, regardless of placeholder settings

## [0.0.059] - 2025-04-24

### Changed
- Increased default volume gain from 3.0 dB to 12.0 dB for better audio clarity

## [0.0.058] - 2025-04-24

### Added
- Added `--srt-min-duration` option to filter out segments shorter than specified duration (default: 0.3s)
- Added `--srt-no-speaker` option to remove speaker labels from SRT output
- Added `--confidence-threshold` option to filter out low-confidence transcriptions (default: 0.5)
- Enhanced transcription quality with improved filtering of short segments
- Added Voice Activity Detection (VAD) filtering during transcription

## [0.0.057] - 2025-04-24

### Added
- Interactive selection of segments files when multiple options are available
- Interactive selection of audio files when multiple files exist in the folder
- Improved user experience for continuation feature with multiple files
- Smart filtering of segments files based on speaker count when specified

## [0.0.056] - 2025-04-24

### Improved
- Enhanced continuation feature to auto-detect input audio file from previous run
- Made `--input-audio` optional when using continuation options
- Added automatic retrieval of original input file from version_info.txt
- Updated documentation with examples of the simplified continuation syntax

## [0.0.055] - 2025-04-24

### Added
- Implemented continuation feature to resume processing from specific steps
- Added `--continue-from` option with choices: preprocessing, diarization, srt
- Added `--continue-folder` option to specify output folder from previous run
- Added `--speaker-count` option to override speaker count configuration
- Created comprehensive tests for continuation functionality

## [0.0.054] - 2025-04-24

### Improved
- Added centralized warning filtering system to suppress dependency warnings
- Created comprehensive test for warning filtering
- Filtered torchaudio, speechbrain, NumExpr, and TensorFlow warnings
- Improved test output clarity by suppressing irrelevant warnings

## [0.0.053] - 2025-04-24

### Improved
- Optimized logging levels to improve console output clarity
- Moved technical details to DEBUG level while keeping essential progress at INFO level
- Added clear documentation of recommended logging levels for different message types
- Created comprehensive test for proper logging level usage

## [0.0.052] - 2025-04-23

### Fixed
- Updated tests to match current implementation
- Fixed test_debug_files_only to reflect current logging behavior
- Updated model loading tests to handle multiple model loading attempts
- Ensured all 60 tests pass with the latest changes

## [0.0.051] - 2025-04-23

### Fixed
- Optimized debug file storage to avoid duplicate large WAV files
- Used symbolic links for WAV conversion debug files to save disk space
- Prevented creation of redundant copies of large audio files

## [0.0.050] - 2025-04-23

### Fixed
- Ensured "Starting Audio Toolkit" is the first visible console message
- Moved "Logging initialized" message to DEBUG level

## [0.0.049] - 2025-04-23

### Fixed
- Improved console logging with clearer, more concise progress messages
- Moved setup and detailed information messages to DEBUG level
- Added user-friendly step completion messages ("WAV file created", "Vocal separation completed")
- Simplified file paths in console output to show only filenames, not full paths
- Kept only essential progress information at INFO level

## [0.0.048] - 2025-04-23

### Fixed
- Fixed audio processing workflow to correctly use the converted WAV file for Demucs vocal separation
- Added clearer logging to show which file is being processed at each step
- Ensured proper file handling throughout the preprocessing pipeline

## [0.0.047] - 2025-04-23

### Fixed
- Improved console output format to remove timestamps and module information
- Simplified console logging to only show the message content
- Maintained detailed logging in log files with timestamps and module info

## [0.0.046] - 2025-04-23

### Fixed
- Fixed `--debug-files-only` flag to create debug files while keeping console at INFO level
- Implemented separate log levels for file and console handlers
- Clarified help text for `--debug-files-only` flag
- Ensured debug files are created in output_dir/debug/ with `--debug-files-only`

## [0.0.045] - 2025-04-23

### Added
- Added comprehensive tests for config.yaml structure verification
- Added integration test for model loading with new configuration
- Created test_model_loading.py script to verify model loading

### Fixed
- Updated config.yaml to use the new structured model configuration
- Fixed model loading with proper categorization (diarization, VAD, segmentation)
- Ensured backward compatibility with old configuration format

## [0.0.044] - 2025-04-23

### Added
- Added `--debug-files-only` command-line switch to enable debug logging to files without console output
- Created test suite for the new debug files only functionality

## [0.0.043] - 2025-04-23

### Added
- Added pytest.ini configuration to suppress deprecation warnings
- Improved test output by filtering out third-party library warnings

## [0.0.042] - 2025-04-23

### Added
- Added comprehensive test suite for configuration loading functionality
- Added tests for new model configuration structure
- Added tests for progress bar handling in demucs
- Updated .windsurfrules with enhanced TDD enforcement mechanisms
- Added detailed TDD workflow steps to .windsurfrules
- Added test enforcement mechanisms with pre-commit hooks

### Changed
- Improved configuration structure with dedicated sections for model types (diarization, VAD, segmentation)
- Enhanced diarization.py to work with the new configuration structure
- Updated config.yaml.template with structured model configuration
- Maintained backward compatibility with old configuration format
- Fixed failing tests to properly use skip_diarization flag
- Updated test files to use the new command-line interface

## [0.0.041] - 2025-04-23

### Added
- Added config.yaml file for external configuration
- Implemented Hugging Face token configuration via config file
- Added clear warnings for missing authentication tokens
- Added fallback to environment variables when config is not available

## [0.0.040] - 2025-04-23

### Fixed
- Added missing subprocess import in audio_toolkit.py
- Fixed dependency checking that was causing NameError exceptions
- Ensured proper imports for all system modules

## [0.0.039] - 2025-04-23

### Fixed
- Completely redesigned Demucs progress bar handling for perfect display
- Eliminated all progress bar interception and processing
- Used direct subprocess execution to preserve terminal control sequences
- Removed all custom progress tracking that was interfering with tqdm
- Fixed inconsistent progress bar display with native tqdm rendering

## [0.0.038] - 2025-04-23

### Fixed
- Enhanced FFmpeg dependency checking with specific version detection
- Added detailed checks for libavutil, libavcodec, and libavformat with version numbers
- Improved error messages for missing FFmpeg libraries
- Provided more specific installation instructions for missing dependencies
- Added better logging of FFmpeg library detection results

## [0.0.037] - 2025-04-23

### Fixed
- Completely redesigned progress bar handling using direct stdout passthrough
- Fixed inconsistent progress bar display by letting tqdm manage its own terminal output
- Used sys.stdout.write and flush for proper terminal control sequence handling
- Eliminated all progress bar detection and formatting to preserve native tqdm behavior
- Maintained error detection and highlighting while preserving progress display

## [0.0.036] - 2025-04-23

### Fixed
- Fixed inconsistent progress bar display with improved detection of tqdm formats
- Removed color formatting from progress bars to preserve original tqdm formatting
- Added flush=True to progress output for more reliable terminal updates
- Enhanced progress bar detection to catch all tqdm variants including seconds/s format

## [0.0.035] - 2025-04-23

### Fixed
- Fixed discrepancy between command-line default gain (3.0 dB) and internal default gain (6.0 dB)
- Updated gain adjustment log message to show decimal precision
- Ensured consistent volume gain settings throughout the application

## [0.0.034] - 2025-04-23

### Fixed
- Completely removed all progress percentage print statements
- Eliminated all logging of progress information
- Let native tqdm progress bars display without any interference
- Removed redundant 0% and 100% progress messages
- Maintained error detection and reporting for critical issues

## [0.0.033] - 2025-04-23

### Fixed
- Completely redesigned progress bar handling to preserve tqdm progress bar formatting
- Removed all console output except for progress bars and error messages
- Eliminated milestone percentage prints that were disrupting progress bar display
- Added error detection to highlight and log any errors during processing
- Ensured clean console output with only progress bars visible during processing

## [0.0.032] - 2025-04-23

### Fixed
- Added custom logging filter to completely suppress progress bars from log files
- Improved detection of tqdm progress bar patterns to prevent them from being logged
- Enhanced progress bar handling to show in console but not in log files
- Fixed issue with progress bars appearing in debug logs
- Implemented comprehensive solution for clean log files without progress bar formatting

## [0.0.031] - 2025-04-23

### Fixed
- Improved compression progress reporting to show all 10% increments (10%, 20%, 30%, etc.)
- Redesigned chunk processing to ensure consistent progress updates
- Used smaller chunk size for more frequent and accurate progress reporting
- Added progress tracking to prevent duplicate progress messages
- Ensured all progress information is printed directly to console with color formatting

## [0.0.030] - 2025-04-23

### Added
- Enhanced version_info.txt with comprehensive input audio file details
- Added system information section with Python version and OS details
- Included detailed audio file metadata (duration, channels, sample rate, etc.)
- Added processing flags section to document enabled/disabled features
- Fixed progress bar formatting in log files by filtering out TQDM output

## [0.0.029] - 2025-04-23

### Added
- Added version_info.txt file to each output directory
- Included toolkit version, processing date, and command used
- Added detailed processing parameters (bit depth, sample rate, filter settings, etc.)
- Improved traceability by documenting the exact configuration used for each run
- Enhanced reproducibility by recording all processing parameters

## [0.0.028] - 2025-04-23

### Fixed
- Fixed progress bar formatting by printing directly to console instead of through logging
- Improved visibility of progress updates with colored output
- Ensured compression and Demucs progress bars display correctly
- Maintained debug-level logging for progress information while showing clean output to users
- Separated console output from log file content for better readability

## [0.0.027] - 2025-04-23

### Fixed
- Improved debug file naming with ordered prefixes (01_highpass, 02_lowpass, etc.)
- Fixed issue where debug files could have inconsistent timestamps
- Ensured debug files appear in correct processing order when sorted alphabetically
- Removed redundant timestamps from debug filenames
- Simplified debug file naming pattern for better readability

## [0.0.026] - 2025-04-23

### Added
- Implemented timestamp-based output subdirectories for each run
- Each run now creates a unique directory with format "YYYYMMDD_HHMMSS_filename"
- Prevents overwriting previous outputs and allows comparing different processing configurations
- All output files (processed audio, SRT, segments) now stored in run-specific directories
- Debug files are now stored in a debug subdirectory within each run directory

## [0.0.025] - 2025-04-23

### Added
- Created dedicated FAQ.md file with comprehensive troubleshooting information
- Improved console output with colored text for errors, warnings, and progress updates
- Enhanced dependency checking to print messages directly to console
- Added support for multiple FFmpeg library versions (libavutil.so.56/57/58)
- Updated progress reporting to show real-time updates in the console

## [0.0.024] - 2025-04-23

### Added
- Added dependency checking for FFmpeg libraries at startup
- Added detailed installation instructions for FFmpeg dependencies in README.md
- Updated requirements.txt with specific versions and system dependency notes
- Implemented helpful error messages for missing dependencies

## [0.0.023] - 2025-04-23

### Fixed
- Fixed compatibility issue with the latest PyAnnote speaker diarization API
- Updated how clustering_threshold parameter is set to match the new API requirements
- Resolved error "SpeakerDiarization.apply() got an unexpected keyword argument 'clustering_threshold'"
- Ensured proper generation of .segments files for all speaker counts

## [0.0.022] - 2025-04-23

### Changed
- Made diarization and SRT generation enabled by default for better user experience
- Changed command-line flags from opt-in to opt-out (--skip-diarization, --skip-srt)
- Replaced all print statements with proper logging calls for consistent output
- Improved file path handling for SRT generation

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


