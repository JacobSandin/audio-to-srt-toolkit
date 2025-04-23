import sys
from pydub import AudioSegment, effects
import numpy as np
import scipy.signal
from scipy import fftpack

def apply_notch_filter(audio, center_freq, width, sample_rate=44100):
    """Apply a notch filter to remove a specific frequency band"""
    nyquist = 0.5 * sample_rate
    low = (center_freq - width/2) / nyquist
    high = (center_freq + width/2) / nyquist
    b, a = scipy.signal.butter(4, [low, high], btype='bandstop')
    filtered = scipy.signal.lfilter(b, a, np.array(audio.get_array_of_samples()))
    return audio._spawn(filtered.astype(audio.array_type))

def spectral_gate(audio, threshold_db=-30, sample_rate=44100):
    """Apply a spectral gate to reduce engine noise while preserving speech"""
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Get the FFT of the signal
    fft = fftpack.fft(samples)
    freqs = fftpack.fftfreq(len(samples), 1/sample_rate)
    
    # Convert threshold to linear scale
    threshold = 10 ** (threshold_db / 20)
    
    # Apply spectral gating - reduce magnitude of frequencies below threshold
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    
    # Calculate the average magnitude in the speech range (500-3000 Hz)
    speech_range = (magnitude[(freqs >= 500) & (freqs <= 3000)])
    if len(speech_range) > 0:
        speech_avg = np.mean(speech_range)
        # Set threshold relative to speech average
        gate_threshold = speech_avg * threshold
        
        # Apply gate
        mask = magnitude > gate_threshold
        gated_magnitude = magnitude * mask
        
        # Reconstruct the signal
        gated_fft = gated_magnitude * np.exp(1j * phase)
        gated_samples = np.real(fftpack.ifft(gated_fft)).astype(audio.array_type)
        
        return audio._spawn(gated_samples)
    
    return audio

def preprocess_for_diarization(input_file, output_file):
    """
    Preprocess audio specifically for diarization with KTM 690 engine noise
    """
    print(f"Loading audio: {input_file}")
    audio = AudioSegment.from_file(input_file)
    sample_rate = audio.frame_rate
    
    # Step 1: High-pass filter to remove the lowest rumble
    print("Applying high-pass filter to remove low rumble...")
    b, a = scipy.signal.butter(4, 120 / (0.5 * sample_rate), btype='high')
    samples = np.array(audio.get_array_of_samples())
    filtered = scipy.signal.lfilter(b, a, samples)
    audio = audio._spawn(filtered.astype(audio.array_type))
    
    # Step 2: Apply targeted notch filters for KTM 690 peak frequencies
    print("Applying notch filters at engine noise peaks...")
    audio = apply_notch_filter(audio, 200, 100, sample_rate)  # Target 150-250 Hz range
    audio = apply_notch_filter(audio, 400, 100, sample_rate)  # Target first harmonic
    
    # Step 3: Apply spectral gating to further reduce engine noise
    print("Applying spectral gating to preserve speech...")
    audio = spectral_gate(audio, threshold_db=-25, sample_rate=sample_rate)
    
    # Step 4: Normalize to ensure speech is at a good level
    print("Normalizing audio...")
    audio = effects.normalize(audio)
    
    # Export
    print(f"Saving to {output_file}")
    audio.export(output_file, format="mp3", bitrate="192k")
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_for_diarization.py input.mp3 output.mp3")
        sys.exit(1)
    
    preprocess_for_diarization(sys.argv[1], sys.argv[2])
