#!/usr/bin/env python3
"""
Steel Pan Sound Generator for deepPan

Creates warm, mellow steel pan sounds with configurable parameters.

Usage:
    python generate_sounds.py                    # Generate all with defaults
    python generate_sounds.py --attack 20       # Custom attack time (ms)
    python generate_sounds.py --preset bright   # Use a preset
    python generate_sounds.py C4                # Generate single note
    python generate_sounds.py --list-params     # Show all parameters
"""

import os
import sys
import argparse
import json
import numpy as np
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 1.5

CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration.json"
)

# Note frequency reference (A4 = 440 Hz)
NOTE_FREQUENCIES = {
    'C': 261.63, 'C#': 277.18, 'Db': 277.18,
    'D': 293.66, 'D#': 311.13, 'Eb': 311.13,
    'E': 329.63,
    'F': 349.23, 'F#': 369.99, 'Gb': 369.99,
    'G': 392.00, 'G#': 415.30, 'Ab': 415.30,
    'A': 440.00, 'A#': 466.16, 'Bb': 466.16,
    'B': 493.88,
}

# Note mapping
NOTE_MAP = {
    # Inner Ring (5 notes, 6th octave)
    'I0': ('C#', 6, 'inner'), 'I1': ('E', 6, 'inner'), 'I2': ('D', 6, 'inner'),
    'I3': ('C', 6, 'inner'), 'I4': ('Eb', 6, 'inner'),
    # Central Ring (12 notes, 5th octave)
    'C0': ('F#', 5, 'central'), 'C1': ('B', 5, 'central'), 'C2': ('E', 5, 'central'),
    'C3': ('A', 5, 'central'), 'C4': ('D', 5, 'central'), 'C5': ('G', 5, 'central'),
    'C6': ('C', 5, 'central'), 'C7': ('F', 5, 'central'), 'C8': ('Bb', 5, 'central'),
    'C9': ('Eb', 5, 'central'), 'C10': ('Ab', 5, 'central'), 'C11': ('C#', 5, 'central'),
    # Outer Ring (12 notes, 4th octave)
    'O0': ('F#', 4, 'outer'), 'O1': ('B', 4, 'outer'), 'O2': ('E', 4, 'outer'),
    'O3': ('A', 4, 'outer'), 'O4': ('D', 4, 'outer'), 'O5': ('G', 4, 'outer'),
    'O6': ('C', 4, 'outer'), 'O7': ('F', 4, 'outer'), 'O8': ('Bb', 4, 'outer'),
    'O9': ('Eb', 4, 'outer'), 'O10': ('Ab', 4, 'outer'), 'O11': ('C#', 4, 'outer'),
}

# Semitone offsets for note-to-MIDI conversion
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}


def note_to_midi(name, octave):
    """Convert note name and octave to MIDI note number."""
    semitone = NOTE_TO_SEMITONE.get(name)
    if semitone is None:
        return None
    return (octave + 1) * 12 + semitone


def load_calibration(path=CALIBRATION_PATH):
    """Load per-note calibration data from JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def get_calibrated_params(calibration, midi_note, base_params):
    """Get synthesis parameters for a MIDI note using calibration data.

    Uses calibration data if available, interpolates from nearest
    calibrated notes if not, falls back to base_params.
    """
    if calibration is None:
        return base_params.copy()

    notes = calibration.get('notes', {})
    defaults = calibration.get('defaults', {})
    key = str(midi_note)

    # Start from base_params (user overrides), overlay calibrated values
    params = base_params.copy()

    # Keys that come from calibration
    cal_keys = ['attack', 'decay', 'sustain', 'release', 'sub_bass',
                'fundamental', 'harmonic2', 'harmonic3', 'harmonic4',
                'filter', 'brightness']

    if key in notes:
        for k in cal_keys:
            if k in notes[key]:
                params[k] = notes[key][k]
            elif k in defaults:
                params[k] = defaults[k]
        if 'detune' in defaults:
            params['detune'] = defaults['detune']
        return params

    # Interpolate from nearest calibrated notes
    calibrated_notes = sorted([int(k) for k in notes.keys()])
    if not calibrated_notes:
        return params

    below = [n for n in calibrated_notes if n < midi_note]
    above = [n for n in calibrated_notes if n > midi_note]

    if below and above:
        lo = below[-1]
        hi = above[0]
        t = (midi_note - lo) / (hi - lo)
        lo_params = notes[str(lo)]
        hi_params = notes[str(hi)]
        for k in cal_keys:
            lo_val = lo_params.get(k, defaults.get(k, params.get(k, 0)))
            hi_val = hi_params.get(k, defaults.get(k, params.get(k, 0)))
            params[k] = int(lo_val + t * (hi_val - lo_val))
    elif below:
        for k in cal_keys:
            if k in notes[str(below[-1])]:
                params[k] = notes[str(below[-1])][k]
            elif k in defaults:
                params[k] = defaults[k]
    elif above:
        for k in cal_keys:
            if k in notes[str(above[0])]:
                params[k] = notes[str(above[0])][k]
            elif k in defaults:
                params[k] = defaults[k]

    if 'detune' in defaults:
        params['detune'] = defaults['detune']
    return params


# Default synthesis parameters
DEFAULT_PARAMS = {
    # Envelope (ADSR)
    'attack': 15,       # Attack time in ms
    'decay': 500,       # Decay time in ms
    'sustain': 20,      # Sustain level (0-100%)
    'release': 300,     # Release time in ms

    # Harmonics (relative amplitude 0-100%)
    'fundamental': 100,
    'harmonic2': 30,    # 2nd harmonic (octave)
    'harmonic3': 10,    # 3rd harmonic
    'harmonic4': 5,     # 4th harmonic
    'sub_bass': 20,     # Sub-harmonic (half frequency)

    # Character
    'detune': 2,        # Detune for beating effect (cents)
    'filter': 6000,     # Low-pass filter cutoff (Hz)
    'brightness': 50,   # Overall brightness (0-100)

    # Output
    'duration': 1.5,    # Note duration in seconds
    'volume': 85,       # Output volume (0-100%)
}

# Presets
PRESETS = {
    'default': DEFAULT_PARAMS.copy(),
    'bright': {
        **DEFAULT_PARAMS,
        'attack': 5, 'decay': 300, 'sustain': 10, 'release': 200,
        'fundamental': 80, 'harmonic2': 50, 'harmonic3': 30, 'harmonic4': 20,
        'sub_bass': 10, 'detune': 3, 'filter': 10000, 'brightness': 80,
    },
    'mellow': {
        **DEFAULT_PARAMS,
        'attack': 30, 'decay': 800, 'sustain': 30, 'release': 500,
        'fundamental': 100, 'harmonic2': 15, 'harmonic3': 5, 'harmonic4': 2,
        'sub_bass': 30, 'detune': 1, 'filter': 3000, 'brightness': 30,
    },
    'bell': {
        **DEFAULT_PARAMS,
        'attack': 2, 'decay': 1500, 'sustain': 5, 'release': 1000,
        'fundamental': 70, 'harmonic2': 60, 'harmonic3': 40, 'harmonic4': 30,
        'sub_bass': 5, 'detune': 5, 'filter': 8000, 'brightness': 60,
    },
    'pluck': {
        **DEFAULT_PARAMS,
        'attack': 1, 'decay': 200, 'sustain': 0, 'release': 100,
        'fundamental': 100, 'harmonic2': 40, 'harmonic3': 20, 'harmonic4': 10,
        'sub_bass': 15, 'detune': 0, 'filter': 5000, 'brightness': 50,
    },
}


def get_frequency(note_name: str, octave: int) -> float:
    """Get frequency for a note in a given octave (A4 = 440 Hz)."""
    base_freq = NOTE_FREQUENCIES.get(note_name)
    if base_freq is None:
        raise ValueError(f"Unknown note: {note_name}")
    return base_freq * (2 ** (octave - 4))


def generate_steel_pan_note(frequency, params, sample_rate=SAMPLE_RATE):
    """Generate a steel pan sound with configurable parameters."""
    duration = params['duration']
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Convert parameters
    attack_time = params['attack'] / 1000
    decay_time = params['decay'] / 1000
    sustain_level = params['sustain'] / 100
    release_time = params['release'] / 1000
    detune_cents = params['detune']

    # Partial structure with configurable amplitudes
    partials = [
        (0.50, params['sub_bass'] / 100, 1.0),           # Sub-harmonic
        (1.00 - detune_cents/1000, params['fundamental'] / 100 * 0.1, 1.0),  # Detuned -
        (1.00, params['fundamental'] / 100, 1.0),        # Fundamental
        (1.00 + detune_cents/1000, params['fundamental'] / 100 * 0.1, 1.0),  # Detuned +
        (2.00, params['harmonic2'] / 100, 1.3),          # 2nd harmonic
        (3.00, params['harmonic3'] / 100, 1.8),          # 3rd harmonic
        (4.00, params['harmonic4'] / 100, 2.2),          # 4th harmonic
    ]

    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    release_samples = int(release_time * sample_rate)

    sound = np.zeros_like(t)

    for ratio, amp, decay_mult in partials:
        if amp < 0.01:
            continue

        freq = frequency * ratio
        if freq > sample_rate / 2 - 500 or freq < 30:
            continue

        # ADSR envelope
        env = np.ones_like(t)

        # Attack
        if attack_samples > 0:
            attack = np.sin(np.linspace(0, np.pi/2, attack_samples)) ** 1.5
            env[:attack_samples] = attack

        # Decay to sustain
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, len(t))
        if decay_end > decay_start:
            decay = np.linspace(1.0, sustain_level, decay_end - decay_start)
            env[decay_start:decay_end] = decay

        # Sustain (exponential decay)
        sustain_start = decay_end
        sustain_tau = decay_time / decay_mult
        if sustain_start < len(t):
            env[sustain_start:] = sustain_level * np.exp(
                -(t[sustain_start:] - t[sustain_start]) / sustain_tau
            )

        # Random phase for natural sound
        phase = np.random.uniform(0, 2 * np.pi)
        partial_sound = amp * env * np.sin(2 * np.pi * freq * t + phase)
        sound += partial_sound

    # Add gentle attack bloom
    bloom_len = int(0.008 * sample_rate)
    if bloom_len > 0:
        bloom_t = np.linspace(0, 0.008, bloom_len)
        bloom = np.sin(2 * np.pi * frequency * bloom_t)
        bloom *= np.exp(-bloom_t * 200)
        bloom *= 0.08
        sound[:bloom_len] += bloom

    # Low-pass filter
    nyq = sample_rate / 2
    cutoff = min(params['filter'], nyq - 100) / nyq
    if 0.01 < cutoff < 0.99:
        b, a = signal.butter(2, cutoff, btype='low')
        sound = signal.filtfilt(b, a, sound)

    # Brightness adjustment (high-shelf)
    if params['brightness'] != 50:
        brightness_boost = (params['brightness'] - 50) / 50
        if brightness_boost > 0:
            # Boost highs
            high_cutoff = 2000 / nyq
            if high_cutoff < 0.99:
                b, a = signal.butter(1, high_cutoff, btype='high')
                highs = signal.filtfilt(b, a, sound)
                sound += highs * brightness_boost * 0.5

    # Soft limiting
    sound = np.tanh(sound * 1.2) / 1.2

    # Normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound = sound / max_val * (params['volume'] / 100)

    return sound


def save_wav(filename, audio, sample_rate=SAMPLE_RATE):
    """Save audio as stereo WAV file."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.82

    audio_int = (audio * 32767).astype(np.int16)
    stereo = np.column_stack((audio_int, audio_int))
    wavfile.write(filename, sample_rate, stereo)


def find_note_by_name(note_str):
    """Find note ID by name like 'C4' or 'F#5'."""
    # Parse note name and octave
    if len(note_str) >= 2:
        if len(note_str) >= 3 and note_str[1] in '#b':
            note_name = note_str[:2]
            octave = int(note_str[2:])
        else:
            note_name = note_str[0].upper()
            octave = int(note_str[1:])

        # Handle enharmonics
        if note_name == 'Db':
            note_name = 'C#'
        elif note_name == 'Gb':
            note_name = 'F#'
        elif note_name == 'D#':
            note_name = 'Eb'
        elif note_name == 'G#':
            note_name = 'Ab'
        elif note_name == 'A#':
            note_name = 'Bb'

        for idx, (n, o, ring) in NOTE_MAP.items():
            if n == note_name and o == octave:
                return idx
    return None


def generate_all(params, output_dir="sounds", verbose=True, calibration=None):
    """Generate all notes with given parameters.

    If calibration data is provided, per-note parameters are used
    (from real sample analysis), with params as the base/fallback.
    """
    os.makedirs(output_dir, exist_ok=True)

    using_cal = calibration is not None
    if verbose:
        print("=" * 60)
        print("Steel Pan Sound Generator - deepPan")
        print("=" * 60)
        if using_cal:
            cal_notes = len(calibration.get('notes', {}))
            print(f"\nUsing per-note calibration ({cal_notes} calibrated notes)")
        print(f"\nBase parameters:")
        print(f"  Envelope: attack={params['attack']}ms, decay={params['decay']}ms, "
              f"sustain={params['sustain']}%, release={params['release']}ms")
        print(f"  Harmonics: fund={params['fundamental']}%, h2={params['harmonic2']}%, "
              f"h3={params['harmonic3']}%, h4={params['harmonic4']}%, sub={params['sub_bass']}%")
        print(f"  Character: detune={params['detune']}¢, filter={params['filter']}Hz, "
              f"brightness={params['brightness']}%")
        print()

    generated = []

    for ring_type in ['inner', 'central', 'outer']:
        if verbose:
            ring_name = {'inner': 'INNER (6ths)', 'central': 'CENTRAL (5ths)',
                        'outer': 'OUTER (4ths)'}[ring_type]
            print(f"\n{ring_name}:")

        for idx, (note_name, octave, nt) in NOTE_MAP.items():
            if nt != ring_type:
                continue

            frequency = get_frequency(note_name, octave)

            # Use per-note calibration if available
            if using_cal:
                midi = note_to_midi(note_name, octave)
                note_params = get_calibrated_params(calibration, midi, params)
                # Ensure duration covers the full decay + release tail
                min_dur = (note_params['attack'] + note_params['decay'] + note_params['release']) / 1000 * 1.2
                note_params['duration'] = max(note_params['duration'], min_dur, 2.5)
            else:
                note_params = params

            sound = generate_steel_pan_note(frequency, note_params)

            safe_name = note_name.replace('#', 's')
            filename = f"{idx}_{safe_name}{octave}.wav"
            filepath = os.path.join(output_dir, filename)

            save_wav(filepath, sound)
            generated.append((idx, note_name, octave, filename))

            if verbose:
                if using_cal:
                    print(f"  {idx:4s}: {note_name:2s}{octave} ({frequency:7.1f} Hz) "
                          f"[A={note_params['attack']} D={note_params['decay']} "
                          f"h2={note_params['harmonic2']} filt={note_params['filter']}] "
                          f"-> {filename}")
                else:
                    print(f"  {idx:4s}: {note_name:2s}{octave} ({frequency:7.1f} Hz) -> {filename}")

    if verbose:
        print()
        print("=" * 60)
        print(f"Generated {len(generated)} sound files in '{output_dir}/'")
        if using_cal:
            print("(using per-note calibration from calibration.json)")
        print("=" * 60)

    # Save parameters used
    params_file = os.path.join(output_dir, "params.json")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    return generated


def main():
    parser = argparse.ArgumentParser(
        description='Generate steel pan sounds with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets: default, bright, mellow, bell, pluck

Examples:
    python generate_sounds.py                      # All notes, default params
    python generate_sounds.py --preset bright      # All notes, bright preset
    python generate_sounds.py --attack 5 --decay 300  # Custom envelope
    python generate_sounds.py C4                   # Single note
    python generate_sounds.py --list-params        # Show all parameters
    python generate_sounds.py --params-file my_sound.json  # Load from file
        """
    )

    # Note selection
    parser.add_argument('note', nargs='?', help='Single note to generate (e.g., C4, F#5)')

    # Preset
    parser.add_argument('--preset', choices=PRESETS.keys(),
                        help='Use a preset configuration')
    parser.add_argument('--params-file', metavar='FILE',
                        help='Load parameters from JSON file (e.g., from synth_pygame.py)')

    # Envelope parameters
    parser.add_argument('--attack', type=int, metavar='MS',
                        help=f'Attack time in ms (default: {DEFAULT_PARAMS["attack"]})')
    parser.add_argument('--decay', type=int, metavar='MS',
                        help=f'Decay time in ms (default: {DEFAULT_PARAMS["decay"]})')
    parser.add_argument('--sustain', type=int, metavar='%',
                        help=f'Sustain level 0-100 (default: {DEFAULT_PARAMS["sustain"]})')
    parser.add_argument('--release', type=int, metavar='MS',
                        help=f'Release time in ms (default: {DEFAULT_PARAMS["release"]})')

    # Harmonics
    parser.add_argument('--fundamental', type=int, metavar='%',
                        help=f'Fundamental amplitude 0-100 (default: {DEFAULT_PARAMS["fundamental"]})')
    parser.add_argument('--harmonic2', type=int, metavar='%',
                        help=f'2nd harmonic amplitude (default: {DEFAULT_PARAMS["harmonic2"]})')
    parser.add_argument('--harmonic3', type=int, metavar='%',
                        help=f'3rd harmonic amplitude (default: {DEFAULT_PARAMS["harmonic3"]})')
    parser.add_argument('--harmonic4', type=int, metavar='%',
                        help=f'4th harmonic amplitude (default: {DEFAULT_PARAMS["harmonic4"]})')
    parser.add_argument('--sub-bass', type=int, metavar='%', dest='sub_bass',
                        help=f'Sub-bass amplitude (default: {DEFAULT_PARAMS["sub_bass"]})')

    # Character
    parser.add_argument('--detune', type=int, metavar='CENTS',
                        help=f'Detune for beating effect (default: {DEFAULT_PARAMS["detune"]})')
    parser.add_argument('--filter', type=int, metavar='HZ',
                        help=f'Low-pass filter cutoff (default: {DEFAULT_PARAMS["filter"]})')
    parser.add_argument('--brightness', type=int, metavar='%',
                        help=f'Brightness 0-100 (default: {DEFAULT_PARAMS["brightness"]})')

    # Output
    parser.add_argument('--duration', type=float, metavar='SEC',
                        help=f'Note duration (default: {DEFAULT_PARAMS["duration"]})')
    parser.add_argument('--volume', type=int, metavar='%',
                        help=f'Output volume 0-100 (default: {DEFAULT_PARAMS["volume"]})')
    parser.add_argument('-o', '--output', default='sounds',
                        help='Output directory (default: sounds)')

    # Calibration
    parser.add_argument('--no-calibration', action='store_true',
                        help='Disable per-note calibration (use uniform params)')
    parser.add_argument('--calibration', metavar='FILE', default=CALIBRATION_PATH,
                        help='Path to calibration.json')

    # Utilities
    parser.add_argument('--list-params', action='store_true',
                        help='List all parameters and exit')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    # List parameters
    if args.list_params:
        print("Sound Generation Parameters:")
        print("\nEnvelope (ADSR):")
        print("  --attack MS      Attack time (1-200ms)")
        print("  --decay MS       Decay time (50-2000ms)")
        print("  --sustain %      Sustain level (0-100%)")
        print("  --release MS     Release time (50-2000ms)")
        print("\nHarmonics:")
        print("  --fundamental %  Fundamental amplitude (0-100%)")
        print("  --harmonic2 %    2nd harmonic (0-100%)")
        print("  --harmonic3 %    3rd harmonic (0-100%)")
        print("  --harmonic4 %    4th harmonic (0-100%)")
        print("  --sub-bass %     Sub-harmonic (0-100%)")
        print("\nCharacter:")
        print("  --detune CENTS   Detuning for beating (0-20)")
        print("  --filter HZ      Low-pass cutoff (500-10000Hz)")
        print("  --brightness %   Overall brightness (0-100%)")
        print("\nOutput:")
        print("  --duration SEC   Note duration (0.5-3.0s)")
        print("  --volume %       Output volume (0-100%)")
        print("\nPresets:", ", ".join(PRESETS.keys()))
        print("\nDefault values:")
        for k, v in DEFAULT_PARAMS.items():
            print(f"  {k}: {v}")
        return

    # Build parameters
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                loaded = json.load(f)
            params = DEFAULT_PARAMS.copy()
            # Map JSON keys to our param names (handle both formats)
            key_map = {
                'sub_bass': 'sub_bass',
                'subBass': 'sub_bass',
                'harmonic2': 'harmonic2',
                'harm2': 'harmonic2',
                'harmonic3': 'harmonic3',
                'harm3': 'harmonic3',
                'harmonic4': 'harmonic4',
                'harm4': 'harmonic4',
            }
            for key, value in loaded.items():
                mapped_key = key_map.get(key, key)
                if mapped_key in params:
                    params[mapped_key] = value
            if not args.quiet:
                print(f"Loaded parameters from: {args.params_file}")
        except Exception as e:
            print(f"Error loading params file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.preset:
        params = PRESETS[args.preset].copy()
    else:
        params = DEFAULT_PARAMS.copy()

    # Override with command-line args
    for key in DEFAULT_PARAMS.keys():
        arg_key = key.replace('_', '-') if '_' in key else key
        arg_val = getattr(args, key.replace('-', '_'), None)
        if arg_val is not None:
            params[key] = arg_val

    # Load calibration data (per-note parameters from real samples)
    calibration = None
    if not args.no_calibration:
        calibration = load_calibration(args.calibration)
        if calibration and not args.quiet:
            cal_notes = len(calibration.get('notes', {}))
            print(f"Loaded calibration data: {cal_notes} notes from {args.calibration}")

    # Generate single note or all
    if args.note:
        note_id = find_note_by_name(args.note)
        if not note_id:
            print(f"Error: Unknown note '{args.note}'", file=sys.stderr)
            print("Available: C4-B4 (outer), C5-B5 (central), C6-Eb6 (inner)")
            sys.exit(1)

        note_name, octave, ring = NOTE_MAP[note_id]
        frequency = get_frequency(note_name, octave)

        # Use per-note calibration if available
        if calibration:
            midi = note_to_midi(note_name, octave)
            note_params = get_calibrated_params(calibration, midi, params)
            min_dur = (note_params['attack'] + note_params['decay'] + note_params['release']) / 1000 * 1.2
            note_params['duration'] = max(note_params['duration'], min_dur, 2.5)
        else:
            note_params = params

        if not args.quiet:
            print(f"Generating {note_name}{octave} ({frequency:.1f} Hz)...")

        sound = generate_steel_pan_note(frequency, note_params)

        os.makedirs(args.output, exist_ok=True)
        safe_name = note_name.replace('#', 's')
        filename = f"{note_id}_{safe_name}{octave}.wav"
        filepath = os.path.join(args.output, filename)

        save_wav(filepath, sound)
        if not args.quiet:
            print(f"Saved: {filepath}")
    else:
        generate_all(params, args.output, verbose=not args.quiet,
                     calibration=calibration)


if __name__ == "__main__":
    main()
