#!/usr/bin/env python3
"""
Steel Pan Sound Generator for deepPan

Creates warm, mellow steel pan sounds for each of the 29 note pads.
Based on the synthesizer from the panned project.

Note Layout (from 3D model analysis):
- Inner Ring (5 notes, 6ths): C#6, E6, D6, C6, Eb6
- Central Ring (12 notes, 5ths): F#5, B5, E5, A5, D5, G5, C5, F5, Bb5, Eb5, Ab5, C#5
- Outer Ring (12 notes, 4ths): F#4, B4, E4, A4, D4, G4, C4, F4, Bb4, Eb4, Ab4, C#4
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 1.5  # seconds

# Note frequency reference (A4 = 440 Hz)
NOTE_FREQUENCIES = {
    'C': 261.63, 'C#': 277.18,
    'D': 293.66, 'Eb': 311.13,
    'E': 329.63,
    'F': 349.23, 'F#': 369.99,
    'G': 392.00, 'Ab': 415.30,
    'A': 440.00, 'Bb': 466.16,
    'B': 493.88,
}

# Note mapping based on 3D model analysis (clockwise from top)
# Format: index -> (note_name, octave, ring_type)
#
# Notes arranged clockwise from 12 o'clock:
# Outer/Central: F#, B, E, A, D, G, C, F, Bb, Eb, Ab, C#
# Inner: C#, E, D, C, Eb
NOTE_MAP = {
    # Inner Ring (5 notes, 6th octave) - clockwise from top
    'I0': ('C#', 6, 'inner'),
    'I1': ('E', 6, 'inner'),
    'I2': ('D', 6, 'inner'),
    'I3': ('C', 6, 'inner'),
    'I4': ('Eb', 6, 'inner'),

    # Central Ring (12 notes, 5th octave) - clockwise from top
    'C0': ('F#', 5, 'central'),
    'C1': ('B', 5, 'central'),
    'C2': ('E', 5, 'central'),
    'C3': ('A', 5, 'central'),
    'C4': ('D', 5, 'central'),
    'C5': ('G', 5, 'central'),
    'C6': ('C', 5, 'central'),
    'C7': ('F', 5, 'central'),
    'C8': ('Bb', 5, 'central'),
    'C9': ('Eb', 5, 'central'),
    'C10': ('Ab', 5, 'central'),
    'C11': ('C#', 5, 'central'),

    # Outer Ring (12 notes, 4th octave) - clockwise from top
    'O0': ('F#', 4, 'outer'),
    'O1': ('B', 4, 'outer'),
    'O2': ('E', 4, 'outer'),
    'O3': ('A', 4, 'outer'),
    'O4': ('D', 4, 'outer'),
    'O5': ('G', 4, 'outer'),
    'O6': ('C', 4, 'outer'),
    'O7': ('F', 4, 'outer'),
    'O8': ('Bb', 4, 'outer'),
    'O9': ('Eb', 4, 'outer'),
    'O10': ('Ab', 4, 'outer'),
    'O11': ('C#', 4, 'outer'),
}


def get_frequency(note_name: str, octave: int) -> float:
    """Get frequency for a note in a given octave (A4 = 440 Hz)."""
    base_freq = NOTE_FREQUENCIES.get(note_name)
    if base_freq is None:
        raise ValueError(f"Unknown note: {note_name}")
    return base_freq * (2 ** (octave - 4))


def generate_steel_pan_note(frequency, duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Generate a soft, warm steel pan sound.

    Key characteristics:
    - Strong fundamental, weaker upper partials
    - Gentle beating from slightly detuned partials
    - Smooth attack (no harsh transient)
    - Low-pass filtering for warmth
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Partial structure: (frequency_ratio, amplitude, decay_multiplier)
    partials = [
        # Subharmonic (gentle warmth)
        (0.50, 0.20, 1.0),

        # Fundamental region - subtle beating
        (0.98, 0.12, 1.0),
        (1.00, 1.00, 1.0),     # Fundamental (dominant)
        (1.02, 0.10, 1.0),

        # Octave region
        (1.99, 0.25, 1.3),
        (2.00, 0.30, 1.3),
        (2.01, 0.08, 1.3),

        # Upper partials - reduced for mellow sound
        (3.00, 0.08, 1.8),
        (4.00, 0.06, 2.2),
        (5.00, 0.03, 2.8),
    ]

    # Envelope parameters
    attack_time = 0.015  # 15ms smooth attack
    attack_samples = int(attack_time * sample_rate)
    base_decay_tau = 0.50  # Decay time constant

    sound = np.zeros_like(t)

    for ratio, amp, decay_mult in partials:
        freq = frequency * ratio

        # Skip frequencies outside valid range
        if freq > sample_rate / 2 - 500 or freq < 50:
            continue

        # Build envelope
        env = np.ones_like(t)

        # Smooth attack curve
        if attack_samples > 0:
            attack = np.sin(np.linspace(0, np.pi/2, attack_samples)) ** 1.5
            env[:attack_samples] = attack

        # Exponential decay
        partial_tau = base_decay_tau / decay_mult
        env[attack_samples:] *= np.exp(-(t[attack_samples:] - t[attack_samples]) / partial_tau)

        # Random phase for natural sound
        phase = np.random.uniform(0, 2 * np.pi)

        partial_sound = amp * env * np.sin(2 * np.pi * freq * t + phase)
        sound += partial_sound

    # Add gentle "bloom" at attack
    bloom_len = int(0.008 * sample_rate)
    if bloom_len > 0:
        bloom_t = np.linspace(0, 0.008, bloom_len)
        bloom = np.sin(2 * np.pi * frequency * bloom_t)
        bloom *= np.exp(-bloom_t * 200)
        bloom *= 0.08
        sound[:bloom_len] += bloom

    # Low-pass filter for warmth
    nyq = sample_rate / 2
    cutoff = min(frequency * 6, 8000) / nyq
    if 0.01 < cutoff < 0.99:
        b, a = signal.butter(2, cutoff, btype='low')
        sound = signal.filtfilt(b, a, sound)

    # Soft limiting
    sound = np.tanh(sound * 1.2) / 1.2

    # Normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound = sound / max_val * 0.85

    return sound


def save_wav(filename, audio, sample_rate=SAMPLE_RATE):
    """Save audio as stereo WAV file."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.82

    audio_int = (audio * 32767).astype(np.int16)
    stereo = np.column_stack((audio_int, audio_int))
    wavfile.write(filename, sample_rate, stereo)


def main():
    output_dir = "sounds"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Steel Pan Sound Generator - deepPan")
    print("=" * 60)
    print()
    print("Generating warm, mellow steel pan sounds...")
    print()

    generated_files = []

    for ring_type in ['inner', 'central', 'outer']:
        ring_name = {'inner': 'INNER (6ths)', 'central': 'CENTRAL (5ths)', 'outer': 'OUTER (4ths)'}[ring_type]
        print(f"\n{ring_name}:")

        for idx, (note_name, octave, nt) in NOTE_MAP.items():
            if nt != ring_type:
                continue

            frequency = get_frequency(note_name, octave)
            sound = generate_steel_pan_note(frequency)

            # URL-safe filename
            safe_name = note_name.replace('#', 's')
            filename = f"{idx}_{safe_name}{octave}.wav"
            filepath = os.path.join(output_dir, filename)

            save_wav(filepath, sound)
            generated_files.append((idx, note_name, octave, filename))

            print(f"  {idx:4s}: {note_name:2s}{octave} ({frequency:7.1f} Hz) -> {filename}")

    print()
    print("=" * 60)
    print(f"Generated {len(generated_files)} sound files in '{output_dir}/'")
    print("=" * 60)

    return generated_files


if __name__ == "__main__":
    main()
