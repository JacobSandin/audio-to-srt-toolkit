#!/bin/bash
# Script to clean up the pyannote_2_1_1 environment and related files

echo "Cleaning up pyannote 2.1.1 related files..."

# Remove the conda environment
echo "Removing conda environment pyannote_2_1_1..."
conda remove -y --name pyannote_2_1_1 --all

# Remove the setup and related files
echo "Removing setup files..."
rm -f setup_pyannote_env.sh
rm -f downgrade_pyannote.sh
rm -f diarization_v2.py
rm -f diarization_properties.py

echo "Cleanup complete!"
echo "You're now using only the optimized pyannote.audio 3.1.1 version with:"
echo "- GPU optimization"
echo "- Progress hooks"
echo "- Advanced speaker separation for Swedish dialects"
