# Frequently Asked Questions

## FFmpeg Dependency Issues

### Q: I'm getting an error about missing FFmpeg libraries even though FFmpeg is installed

If you see errors like:
```
OSError: libavutil.so.58: cannot open shared object file: No such file or directory
```
or
```
Missing FFmpeg libraries. Please install them with:
sudo apt-get install ffmpeg libavutil-dev libavcodec-dev libavformat-dev
```

This happens because:
1. The FFmpeg command is installed but the development libraries are missing
2. The FFmpeg libraries are installed but in a non-standard location
3. There's a version mismatch between the installed libraries and what the Python packages expect

**Solutions:**

1. **Install the development libraries:**
   ```bash
   sudo apt-get install libavutil-dev libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev libswresample-dev
   ```

2. **Create symbolic links if library versions don't match:**
   If you have `libavutil.so.57` but the code is looking for `libavutil.so.58`, you can create a symbolic link:
   ```bash
   # Find your actual library location
   find /usr -name "libavutil.so*"
   
   # Create a symbolic link (adjust paths as needed)
   sudo ln -s /usr/lib/x86_64-linux-gnu/libavutil.so.57 /usr/lib/x86_64-linux-gnu/libavutil.so.58
   ```

3. **Set the library path environment variable:**
   ```bash
   export LD_LIBRARY_PATH=/path/to/your/ffmpeg/lib:$LD_LIBRARY_PATH
   ```

4. **Modify the dependency check in the code:**
   The toolkit now tries multiple library versions, but you can edit `audio_toolkit.py` to add your specific library version if needed.

### Q: How do I check which FFmpeg libraries I have installed?

Use these commands to check your FFmpeg installation:

```bash
# Check FFmpeg version
ffmpeg -version

# Find installed FFmpeg libraries
find /usr -name "libav*.so*"

# Check which libraries a Python package is trying to load
LD_DEBUG=libs python -c "import torio; print('torio imported successfully')"
```

## Diarization Issues

### Q: The diarization is not producing any .segments files

This could be due to:
1. Missing Hugging Face authentication token
2. Incompatible PyAnnote version
3. Insufficient speaker data in the audio

**Solutions:**
1. Set your Hugging Face token: `export HF_TOKEN=your_token_here`
2. Make sure you're using PyAnnote 3.1.0 or later
3. Try adjusting the min/max speaker parameters: `--min-speakers 2 --max-speakers 4`

## Audio Processing

### Q: What are the optimal filter settings for Swedish dialects?

Based on extensive testing:
- Highpass filter: 3500-4000Hz (default: 3750Hz)
- Lowpass filter: 5500-8000Hz (default: 8000Hz)

Higher highpass cutoffs (3000-4000Hz) significantly improve speech clarity and dialect distinction for Swedish speakers.

## Performance Issues

### Q: The processing is very slow, how can I speed it up?

1. Enable GPU acceleration if available
2. Reduce audio quality (use 16-bit instead of 24-bit)
3. Process shorter audio segments (split long recordings)
4. Skip preprocessing if you've already processed the audio: `--skip-preprocessing`
