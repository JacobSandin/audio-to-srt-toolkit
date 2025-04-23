import sys
from pydub import AudioSegment, effects
import numpy as np
import scipy.signal
import noisereduce as nr

def highpass_filter(audio, cutoff=150, sample_rate=44100):
    b, a = scipy.signal.butter(2, cutoff / (0.5 * sample_rate), btype='high')
    filtered = scipy.signal.lfilter(b, a, np.array(audio.get_array_of_samples()))
    return audio._spawn(filtered.astype(audio.array_type))

def lowpass_filter(audio, cutoff=8000, sample_rate=44100):
    nyquist = 0.5 * sample_rate
    if cutoff >= nyquist:
        print(f"[WARNING] Low-pass cutoff {cutoff} Hz is >= Nyquist ({nyquist} Hz). Lowering cutoff to {nyquist - 100} Hz.")
        cutoff = nyquist - 100  # Leave a small margin
    b, a = scipy.signal.butter(2, cutoff / nyquist, btype='low')
    filtered = scipy.signal.lfilter(b, a, np.array(audio.get_array_of_samples()))
    return audio._spawn(filtered.astype(audio.array_type))

def apply_noise_reduction(audio, sample_rate=44100):
    data = np.array(audio.get_array_of_samples())
    reduced = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        prop_decrease=0.2,  # less aggressive than default
        stationary=False,    # if your noise is constant
        n_std_thresh_stationary=1.0  # less aggressive threshold
    )
    return audio._spawn(reduced.astype(audio.array_type))

def preprocess_audio(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    sample_rate = audio.frame_rate

    # Step 1: High-pass filter (remove engine rumble)
    audio = highpass_filter(audio, cutoff=150, sample_rate=sample_rate)
    # Step 2: Low-pass filter (remove high wind)
    audio = lowpass_filter(audio, cutoff=8000, sample_rate=sample_rate)
    # Step 3: Gentler compression (threshold -10.0 dB, ratio 2.0)
    audio = effects.compress_dynamic_range(audio, threshold=-10.0, ratio=2.0)
    # Step 4: Noise reduction DISABLED for clarity
    audio = apply_noise_reduction(audio, sample_rate=sample_rate)
    # Step 5: Normalize after all processing
    audio = effects.normalize(audio)
    # Step 6: Optional gain boost (+3 dB)
    audio += 6  # Increase volume by 3 dB; adjust as needed
    # Step 7: Export at higher quality (192k bitrate)
    audio.export(output_file, format="mp3", bitrate="192k")
    print(f"Preprocessed audio saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_audio.py input.mp3 output.mp3")
        sys.exit(1)
    preprocess_audio(sys.argv[1], sys.argv[2])
