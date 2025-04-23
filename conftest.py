#!/usr/bin/env python3
# pytest configuration file
# Configures pytest behavior and filters warnings
# 2025-04-23 - JS

import warnings

# Filter out warnings from dependencies
warnings.filterwarnings("ignore", message="torchaudio._backend.*has been deprecated")
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
warnings.filterwarnings("ignore", message="'audioop' is deprecated and slated for removal")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydub.utils")
warnings.filterwarnings("ignore", message="The get_cmap function was deprecated in Matplotlib 3.7")
warnings.filterwarnings("ignore", message="`torchaudio.backend.common.AudioMetaData` has been moved to `torchaudio.AudioMetaData`")
