#!/usr/bin/env python3
"""
Audio analysis module for deepPan.

Analyzes audio files to:
1. Detect notes being played (pitch detection)
2. Extract harmonic content
3. Estimate ADSR envelope
4. Generate synth parameters to match the sound

Example usage:
    python analyze_audio.py sample.wav
    python analyze_audio.py sample.wav --output params.json
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, fftfreq
import json
import argparse

# Note frequencies (A4 = 440 Hz)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Tenor pan notes
TENOR_PAN_NOTES = [
    'F#4', 'B4', 'E4', 'A4', 'D4', 'G4', 'C4', 'F4', 'Bb4', 'Eb4', 'Ab4', 'C#4',
    'F#5', 'B5', 'E5', 'A5', 'D5', 'G5', 'C5', 'F5', 'Bb5', 'Eb5', 'Ab5', 'C#5',
    'C#6', 'E6', 'D6', 'C6', 'Eb6'
]


def freq_to_note(freq):
    """Convert frequency to note name and octave."""
    if freq <= 0:
        return None, None
    # A4 = 440 Hz is MIDI note 69
    midi_note = 69 + 12 * np.log2(freq / 440.0)
    midi_note = int(round(midi_note))
    octave = (midi_note // 12) - 1
    note_idx = midi_note % 12
    note_name = NOTE_NAMES[note_idx]
    return note_name, octave


def note_to_freq(note_str):
    """Convert note string like 'C4' or 'F#5' to frequency."""
    # Handle flats
    note_str = note_str.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#')
    note_str = note_str.replace('Ab', 'G#').replace('Bb', 'A#')

    if len(note_str) >= 2:
        if len(note_str) >= 3 and note_str[1] == '#':
            note_name = note_str[:2]
            octave = int(note_str[2:])
        else:
            note_name = note_str[0]
            octave = int(note_str[1:])

        note_idx = NOTE_NAMES.index(note_name)
        midi_note = (octave + 1) * 12 + note_idx
        return 440.0 * (2 ** ((midi_note - 69) / 12))
    return 440.0


def detect_pitch(audio, sample_rate, min_freq=50, max_freq=2000):
    """
    Detect the fundamental frequency using autocorrelation.
    Returns frequency in Hz.
    """
    # Use a segment from the attack/sustain portion
    segment_start = int(0.05 * sample_rate)  # Skip first 50ms
    segment_length = int(0.2 * sample_rate)   # Analyze 200ms
    segment_end = min(segment_start + segment_length, len(audio))

    if segment_end <= segment_start:
        segment = audio
    else:
        segment = audio[segment_start:segment_end]

    # Normalize
    segment = segment - np.mean(segment)
    if np.max(np.abs(segment)) > 0:
        segment = segment / np.max(np.abs(segment))

    # Autocorrelation
    corr = np.correlate(segment, segment, mode='full')
    corr = corr[len(corr)//2:]

    # Find peaks
    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)

    if max_lag > len(corr):
        max_lag = len(corr) - 1

    corr_segment = corr[min_lag:max_lag]
    if len(corr_segment) == 0:
        return 0

    peak_idx = np.argmax(corr_segment) + min_lag

    if peak_idx > 0:
        freq = sample_rate / peak_idx
        return freq
    return 0


def detect_pitch_fft(audio, sample_rate, min_freq=50, max_freq=2000):
    """
    Detect fundamental frequency using FFT with harmonic product spectrum.
    More robust for complex tones.
    """
    # Use middle portion of audio
    segment_start = int(0.05 * sample_rate)
    segment_length = int(0.3 * sample_rate)
    segment_end = min(segment_start + segment_length, len(audio))

    if segment_end <= segment_start:
        segment = audio
    else:
        segment = audio[segment_start:segment_end]

    # Apply window
    window = np.hanning(len(segment))
    segment = segment * window

    # FFT
    n = len(segment)
    spectrum = np.abs(fft(segment))[:n//2]
    freqs = fftfreq(n, 1/sample_rate)[:n//2]

    # Harmonic Product Spectrum
    hps = spectrum.copy()
    num_harmonics = 5

    for h in range(2, num_harmonics + 1):
        decimated = spectrum[::h]
        hps[:len(decimated)] *= decimated

    # Find peak in valid frequency range
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    valid_hps = hps.copy()
    valid_hps[~freq_mask] = 0

    peak_idx = np.argmax(valid_hps)
    if peak_idx > 0 and freqs[peak_idx] >= min_freq:
        return freqs[peak_idx]

    return 0


def analyze_harmonics(audio, sample_rate, fundamental_freq):
    """
    Analyze the harmonic content relative to the fundamental.
    Returns dict with relative amplitudes of harmonics.
    """
    # Use sustain portion
    segment_start = int(0.1 * sample_rate)
    segment_length = int(0.3 * sample_rate)
    segment_end = min(segment_start + segment_length, len(audio))

    if segment_end <= segment_start:
        segment = audio
    else:
        segment = audio[segment_start:segment_end]

    # FFT
    n = len(segment)
    window = np.hanning(n)
    spectrum = np.abs(fft(segment * window))[:n//2]
    freqs = fftfreq(n, 1/sample_rate)[:n//2]

    # Find amplitude at each harmonic
    harmonics = {}
    harmonic_ratios = [0.5, 1.0, 2.0, 3.0, 4.0]  # sub, fund, 2nd, 3rd, 4th
    harmonic_names = ['sub_bass', 'fundamental', 'harmonic2', 'harmonic3', 'harmonic4']

    fund_amp = 0

    for ratio, name in zip(harmonic_ratios, harmonic_names):
        target_freq = fundamental_freq * ratio

        # Find nearest frequency bin (with some tolerance)
        tolerance = fundamental_freq * 0.05  # 5% tolerance
        freq_mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)

        if np.any(freq_mask):
            amp = np.max(spectrum[freq_mask])
            harmonics[name] = amp
            if name == 'fundamental':
                fund_amp = amp
        else:
            harmonics[name] = 0

    # Normalize to fundamental
    if fund_amp > 0:
        for name in harmonic_names:
            harmonics[name] = min(100, int((harmonics[name] / fund_amp) * 100))

    return harmonics


def analyze_envelope(audio, sample_rate):
    """
    Estimate ADSR envelope parameters from audio.
    Returns dict with attack, decay, sustain, release in ms/%.
    """
    # Get amplitude envelope
    envelope = np.abs(audio)

    # Smooth the envelope
    window_size = int(0.01 * sample_rate)  # 10ms window
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        envelope = np.convolve(envelope, kernel, mode='same')

    # Normalize
    max_amp = np.max(envelope)
    if max_amp > 0:
        envelope = envelope / max_amp

    # Find attack time (time to reach 90% of peak)
    peak_idx = np.argmax(envelope)
    attack_threshold = 0.9
    attack_idx = 0
    for i in range(peak_idx):
        if envelope[i] >= attack_threshold:
            attack_idx = i
            break
    attack_ms = (attack_idx / sample_rate) * 1000

    # Find sustain level (average level in middle portion)
    sustain_start = int(0.3 * len(envelope))
    sustain_end = int(0.6 * len(envelope))
    if sustain_end > sustain_start:
        sustain_level = np.mean(envelope[sustain_start:sustain_end])
    else:
        sustain_level = 0.2

    # Find decay time (time from peak to sustain level)
    decay_idx = peak_idx
    for i in range(peak_idx, len(envelope)):
        if envelope[i] <= sustain_level * 1.1:
            decay_idx = i
            break
    decay_ms = ((decay_idx - peak_idx) / sample_rate) * 1000

    # Find release time (time for final portion to decay)
    release_start = int(0.7 * len(envelope))
    if release_start < len(envelope):
        release_segment = envelope[release_start:]
        # Find time to drop to 10% of value at release_start
        if len(release_segment) > 0 and release_segment[0] > 0:
            threshold = release_segment[0] * 0.1
            release_idx = len(release_segment)
            for i, val in enumerate(release_segment):
                if val <= threshold:
                    release_idx = i
                    break
            release_ms = (release_idx / sample_rate) * 1000
        else:
            release_ms = 300
    else:
        release_ms = 300

    return {
        'attack': max(1, min(200, int(attack_ms))),
        'decay': max(50, min(2000, int(decay_ms))),
        'sustain': max(0, min(100, int(sustain_level * 100))),
        'release': max(50, min(2000, int(release_ms)))
    }


def estimate_filter_brightness(audio, sample_rate, fundamental_freq):
    """
    Estimate filter cutoff and brightness from spectral shape.
    """
    # FFT of full audio
    n = len(audio)
    window = np.hanning(n)
    spectrum = np.abs(fft(audio * window))[:n//2]
    freqs = fftfreq(n, 1/sample_rate)[:n//2]

    # Find spectral centroid (brightness indicator)
    if np.sum(spectrum) > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        centroid = 1000

    # Find rolloff frequency (where 85% of energy is below)
    cumsum = np.cumsum(spectrum)
    total = cumsum[-1]
    if total > 0:
        rolloff_idx = np.where(cumsum >= 0.85 * total)[0]
        if len(rolloff_idx) > 0:
            rolloff_freq = freqs[rolloff_idx[0]]
        else:
            rolloff_freq = 6000
    else:
        rolloff_freq = 6000

    # Map to filter and brightness parameters
    filter_cutoff = max(500, min(10000, int(rolloff_freq)))

    # Brightness based on centroid relative to fundamental
    brightness_ratio = centroid / fundamental_freq if fundamental_freq > 0 else 2
    brightness = max(0, min(100, int((brightness_ratio - 1) * 25 + 50)))

    return {
        'filter': filter_cutoff,
        'brightness': brightness
    }


def analyze_audio(filepath):
    """
    Analyze an audio file and extract synthesis parameters.

    Returns:
        dict with:
            - detected_notes: list of (note_name, confidence)
            - on_pan: list of notes that match tenor pan
            - params: synthesis parameters to match the sound
            - analysis: detailed analysis data
    """
    # Load audio
    sample_rate, audio = wavfile.read(filepath)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # Detect fundamental frequency
    freq_autocorr = detect_pitch(audio, sample_rate)
    freq_fft = detect_pitch_fft(audio, sample_rate)

    # Use FFT result if autocorrelation seems off
    if abs(freq_fft - freq_autocorr) > freq_autocorr * 0.1:
        fundamental_freq = freq_fft
    else:
        fundamental_freq = (freq_autocorr + freq_fft) / 2

    # Convert to note
    note_name, octave = freq_to_note(fundamental_freq)
    detected_note = f"{note_name}{octave}" if note_name else "Unknown"

    # Check if note is on tenor pan
    # Handle enharmonic equivalents
    detected_variants = [detected_note]
    if '#' in detected_note:
        # Add flat equivalent
        note_map = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
        base = detected_note[:-1]
        if base in note_map:
            detected_variants.append(note_map[base] + detected_note[-1])
    elif 'b' in detected_note:
        # Add sharp equivalent
        note_map = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
        idx = detected_note.index('b')
        base = detected_note[:idx+1]
        if base in note_map:
            detected_variants.append(note_map[base] + detected_note[idx+1:])

    on_pan = [n for n in detected_variants if n in TENOR_PAN_NOTES]

    # Analyze harmonics
    harmonics = analyze_harmonics(audio, sample_rate, fundamental_freq)

    # Analyze envelope
    envelope = analyze_envelope(audio, sample_rate)

    # Estimate filter/brightness
    filter_brightness = estimate_filter_brightness(audio, sample_rate, fundamental_freq)

    # Combine into synth parameters
    params = {
        'attack': envelope['attack'],
        'decay': envelope['decay'],
        'sustain': envelope['sustain'],
        'release': envelope['release'],
        'fundamental': harmonics.get('fundamental', 100),
        'harmonic2': harmonics.get('harmonic2', 30),
        'harmonic3': harmonics.get('harmonic3', 10),
        'harmonic4': harmonics.get('harmonic4', 5),
        'sub_bass': harmonics.get('sub_bass', 20),
        'detune': 2,  # Hard to detect, use default
        'filter': filter_brightness['filter'],
        'brightness': filter_brightness['brightness'],
        'duration': len(audio) / sample_rate,
        'volume': 85
    }

    return {
        'detected_note': detected_note,
        'detected_frequency': round(fundamental_freq, 2),
        'expected_frequency': round(note_to_freq(detected_note), 2) if note_name else None,
        'on_pan': on_pan,
        'is_pan_note': len(on_pan) > 0,
        'params': params,
        'analysis': {
            'harmonics': harmonics,
            'envelope': envelope,
            'filter_brightness': filter_brightness,
            'duration_seconds': round(len(audio) / sample_rate, 2),
            'sample_rate': sample_rate
        }
    }


def print_analysis(result):
    """Print analysis results in a readable format."""
    print("\n" + "="*50)
    print("AUDIO ANALYSIS RESULTS")
    print("="*50)

    print(f"\nDetected Note: {result['detected_note']}")
    print(f"Detected Frequency: {result['detected_frequency']} Hz")
    if result['expected_frequency']:
        print(f"Expected Frequency: {result['expected_frequency']} Hz")
        cents_off = 1200 * np.log2(result['detected_frequency'] / result['expected_frequency'])
        print(f"Tuning: {cents_off:+.1f} cents")

    print(f"\nOn Tenor Pan: {'Yes - ' + ', '.join(result['on_pan']) if result['is_pan_note'] else 'No'}")

    print("\n--- Envelope (ADSR) ---")
    env = result['analysis']['envelope']
    print(f"Attack:  {env['attack']} ms")
    print(f"Decay:   {env['decay']} ms")
    print(f"Sustain: {env['sustain']}%")
    print(f"Release: {env['release']} ms")

    print("\n--- Harmonics ---")
    harm = result['analysis']['harmonics']
    print(f"Sub Bass:    {harm.get('sub_bass', 0)}%")
    print(f"Fundamental: {harm.get('fundamental', 100)}%")
    print(f"Harmonic 2:  {harm.get('harmonic2', 0)}%")
    print(f"Harmonic 3:  {harm.get('harmonic3', 0)}%")
    print(f"Harmonic 4:  {harm.get('harmonic4', 0)}%")

    print("\n--- Character ---")
    fb = result['analysis']['filter_brightness']
    print(f"Filter:     {fb['filter']} Hz")
    print(f"Brightness: {fb['brightness']}%")

    print(f"\nDuration: {result['analysis']['duration_seconds']} seconds")

    print("\n--- Synth Parameters ---")
    print(json.dumps(result['params'], indent=2))

    print("\n--- CLI Command ---")
    cmd = "python generate_sounds.py"
    defaults = {'attack': 15, 'decay': 500, 'sustain': 20, 'release': 300,
                'fundamental': 100, 'harmonic2': 30, 'harmonic3': 10, 'harmonic4': 5,
                'sub_bass': 20, 'detune': 2, 'filter': 6000, 'brightness': 50,
                'duration': 1.5, 'volume': 85}
    for key, value in result['params'].items():
        if value != defaults.get(key):
            cli_key = key.replace('_', '-')
            cmd += f" --{cli_key} {value}"
    print(cmd)


def print_summary_table(results):
    """Print a summary table of multiple analysis results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    # Header
    print(f"{'File':<30} {'Note':<6} {'Freq':>8} {'On Pan':<6} {'A':>4} {'D':>5} {'S':>4} {'R':>5}")
    print("-"*80)

    for filepath, result in results.items():
        filename = filepath.split('/')[-1][:28]
        note = result['detected_note']
        freq = f"{result['detected_frequency']:.1f}"
        on_pan = "Yes" if result['is_pan_note'] else "No"
        a = result['params']['attack']
        d = result['params']['decay']
        s = result['params']['sustain']
        r = result['params']['release']
        print(f"{filename:<30} {note:<6} {freq:>8} {on_pan:<6} {a:>4} {d:>5} {s:>4} {r:>5}")

    print("-"*80)

    # Count notes on pan
    on_pan_count = sum(1 for r in results.values() if r['is_pan_note'])
    print(f"Total: {len(results)} files, {on_pan_count} notes on tenor pan")

    # Average parameters
    if results:
        avg_params = {}
        param_keys = ['attack', 'decay', 'sustain', 'release', 'fundamental',
                      'harmonic2', 'harmonic3', 'harmonic4', 'sub_bass',
                      'filter', 'brightness']
        for key in param_keys:
            avg_params[key] = int(np.mean([r['params'][key] for r in results.values()]))

        print("\n--- Average Parameters ---")
        print(json.dumps(avg_params, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Analyze audio to extract steel pan synthesis parameters')
    parser.add_argument('audio_files', nargs='+', help='Path to WAV file(s) to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file for parameters (single file) or directory (multiple files)')
    parser.add_argument('--json', action='store_true', help='Output as JSON only')
    parser.add_argument('--summary', '-s', action='store_true', help='Show summary table only (for multiple files)')

    args = parser.parse_args()

    results = {}
    errors = []

    for audio_file in args.audio_files:
        try:
            result = analyze_audio(audio_file)
            results[audio_file] = result

            # For single file or non-summary mode, print full analysis
            if len(args.audio_files) == 1 or not args.summary:
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print_analysis(result)

        except FileNotFoundError:
            errors.append(f"File not found: {audio_file}")
        except Exception as e:
            errors.append(f"Error analyzing {audio_file}: {e}")

    # Print summary table for multiple files
    if len(args.audio_files) > 1 and results:
        print_summary_table(results)

    # Handle output
    if args.output and results:
        import os
        if len(results) == 1:
            # Single file - save params directly
            result = list(results.values())[0]
            with open(args.output, 'w') as f:
                json.dump(result['params'], f, indent=2)
            print(f"\nParameters saved to {args.output}")
        else:
            # Multiple files - save to directory or combined JSON
            if args.output.endswith('.json'):
                # Save all results to single JSON
                all_params = {filepath: r['params'] for filepath, r in results.items()}
                with open(args.output, 'w') as f:
                    json.dump(all_params, f, indent=2)
                print(f"\nAll parameters saved to {args.output}")
            else:
                # Save to directory
                os.makedirs(args.output, exist_ok=True)
                for filepath, result in results.items():
                    basename = os.path.splitext(os.path.basename(filepath))[0]
                    outfile = os.path.join(args.output, f"{basename}_params.json")
                    with open(outfile, 'w') as f:
                        json.dump(result['params'], f, indent=2)
                print(f"\nParameters saved to {args.output}/ ({len(results)} files)")

    # Report errors
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
