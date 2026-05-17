#!/usr/bin/env python3
"""
Steel Pan MIDI Sampler for deepPan

Sample-based steel pan instrument using recordings from urbanPan.
Downloads real steel pan WAV samples and maps them to MIDI notes
with velocity-sensitive layering.

Usage:
    python midi_sampler.py                      # Interactive keyboard mode
    python midi_sampler.py --download           # Download samples only
    python midi_sampler.py --play file.mid      # Play a MIDI file
    python midi_sampler.py --create-demo        # Create and play a demo MIDI file
    python midi_sampler.py --list-midi          # List available MIDI input devices
    python midi_sampler.py --midi-input 0       # Use MIDI controller (device index)

Requirements:
    pip install pygame mido numpy

Optional (for MIDI controller input):
    pip install python-rtmidi
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import numpy as np
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urbanpan_samples")

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/urbansmash/urbanPan/master/urbanPan/Samples"
)

# urbanPan sample naming: {velocity_layer}-{note}{octave}.wav
# Note names use S for sharp (e.g., FS3 = F#3, CS4 = C#4)
# Velocity layers: 0 (soft), 1 (medium), 2 (loud/primary), 3 (very loud, sparse)
# Range: F3 (MIDI 53) to C6 (MIDI 84)

NOTE_NAMES_SHARP = ["C", "CS", "D", "DS", "E", "F", "FS", "G", "GS", "A", "AS", "B"]
NOTE_NAMES_DISPLAY = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

# MIDI note 53 = F3, MIDI note 84 = C6
MIDI_LOW = 53   # F3
MIDI_HIGH = 84  # C6

# Velocity layers available in urbanPan (layer: [notes available])
# Layer 1 and 2 have full chromatic coverage (F3-C6)
# Layer 0 and 3 are sparse
FULL_LAYERS = [1, 2]
SPARSE_LAYERS = [0, 3]

SAMPLE_RATE = 44100


def midi_to_note_name(midi_note):
    """Convert MIDI note number to urbanPan filename format (e.g., 53 -> 'F3')."""
    octave = (midi_note // 12) - 1
    note_idx = midi_note % 12
    return f"{NOTE_NAMES_SHARP[note_idx]}{octave}"


def midi_to_display_name(midi_note):
    """Convert MIDI note number to readable name (e.g., 53 -> 'F3')."""
    octave = (midi_note // 12) - 1
    note_idx = midi_note % 12
    return f"{NOTE_NAMES_DISPLAY[note_idx]}{octave}"


# ---------------------------------------------------------------------------
# Sample downloading
# ---------------------------------------------------------------------------

def get_sample_filename(layer, midi_note):
    """Get the WAV filename for a given velocity layer and MIDI note."""
    note_name = midi_to_note_name(midi_note)
    return f"{layer}-{note_name}.wav"


def download_samples(force=False):
    """Download steel pan samples from urbanPan GitHub repository."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Build list of files to download
    files_to_download = []

    # Full layers (1, 2): all notes F3 to C6
    for layer in FULL_LAYERS:
        for midi_note in range(MIDI_LOW, MIDI_HIGH + 1):
            fname = get_sample_filename(layer, midi_note)
            files_to_download.append(fname)

    # Sparse layers (0, 3): known files from urbanPan repo
    sparse_0_notes = ["A4", "A5", "AS4", "AS5", "B3", "DS5", "E5", "FS5", "G5", "GS3"]
    for note in sparse_0_notes:
        files_to_download.append(f"0-{note}.wav")

    sparse_3_notes = ["B5", "C6", "D5", "GS5"]
    for note in sparse_3_notes:
        files_to_download.append(f"3-{note}.wav")

    # Also grab the SoundFont
    files_to_download.append("urbanPan.sf2")

    # Extra files
    files_to_download.extend(["FN03.wav", "FS03.wav"])

    downloaded = 0
    skipped = 0
    failed = 0

    print(f"Downloading urbanPan samples to {SAMPLE_DIR}/")
    print(f"  {len(files_to_download)} files to check...\n")

    for fname in files_to_download:
        dest = os.path.join(SAMPLE_DIR, fname)
        if os.path.exists(dest) and not force:
            skipped += 1
            continue

        url = f"{GITHUB_RAW_BASE}/{fname}"
        try:
            print(f"  Downloading {fname}...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            downloaded += 1
        except urllib.error.HTTPError as e:
            print(f"FAILED ({e.code})")
            failed += 1
        except urllib.error.URLError as e:
            print(f"ERROR ({e.reason})")
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} already present, {failed} failed")
    return downloaded + skipped > 0


def ensure_samples():
    """Make sure samples are downloaded, download if not present."""
    # Check if we have at least the primary layer (2-series)
    test_file = os.path.join(SAMPLE_DIR, "2-C4.wav")
    if not os.path.exists(test_file):
        print("Steel pan samples not found. Downloading from urbanPan...\n")
        download_samples()
    return os.path.exists(test_file)


# ---------------------------------------------------------------------------
# Sample loading and playback engine
# ---------------------------------------------------------------------------

class SteelPanSampler:
    """Sample-based steel pan instrument with velocity-sensitive playback."""

    def __init__(self, max_polyphony=16):
        import pygame
        import pygame.mixer

        self.pygame = pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

        self.max_polyphony = max_polyphony
        self.samples = {}      # {(layer, midi_note): pygame.mixer.Sound}
        self.channels = []     # Available mixer channels
        self.loaded = False

        # Set up mixer channels for polyphony
        pygame.mixer.set_num_channels(max_polyphony)
        self.channels = [pygame.mixer.Channel(i) for i in range(max_polyphony)]
        self._next_channel = 0

    def load_samples(self):
        """Load all available WAV samples into memory."""
        pygame = self.pygame
        loaded_count = 0

        for layer in [0, 1, 2, 3]:
            for midi_note in range(MIDI_LOW, MIDI_HIGH + 1):
                fname = get_sample_filename(layer, midi_note)
                fpath = os.path.join(SAMPLE_DIR, fname)
                if os.path.exists(fpath):
                    try:
                        sound = pygame.mixer.Sound(fpath)
                        self.samples[(layer, midi_note)] = sound
                        loaded_count += 1
                    except Exception as e:
                        print(f"  Warning: Could not load {fname}: {e}")

        self.loaded = True
        layer_counts = {}
        for (layer, _) in self.samples:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        print(f"Loaded {loaded_count} samples across {len(layer_counts)} velocity layers")
        for layer in sorted(layer_counts):
            print(f"  Layer {layer}: {layer_counts[layer]} notes")

    def _velocity_to_layer(self, velocity):
        """Map MIDI velocity (0-127) to the best available sample layer."""
        # Layer mapping: 0=pp, 1=mp, 2=f, 3=ff
        if velocity < 32:
            return 0
        elif velocity < 64:
            return 1
        elif velocity < 100:
            return 2
        else:
            return 3

    def _get_sample(self, midi_note, velocity=100):
        """Get the best matching sample for a note and velocity."""
        target_layer = self._velocity_to_layer(velocity)

        # Try exact layer first, then fall back to nearest available
        search_order = sorted([0, 1, 2, 3], key=lambda l: abs(l - target_layer))
        for layer in search_order:
            key = (layer, midi_note)
            if key in self.samples:
                return self.samples[key], layer

        return None, None

    def _get_channel(self):
        """Get the next available mixer channel (round-robin with steal)."""
        ch = self.channels[self._next_channel]
        self._next_channel = (self._next_channel + 1) % self.max_polyphony
        return ch

    def note_on(self, midi_note, velocity=100):
        """Play a steel pan note."""
        if midi_note < MIDI_LOW or midi_note > MIDI_HIGH:
            return

        sample, layer = self._get_sample(midi_note, velocity)
        if sample is None:
            return

        # Scale volume by velocity
        vol = (velocity / 127.0) ** 0.7  # Slight curve for more natural feel
        sample.set_volume(vol)

        ch = self._get_channel()
        ch.play(sample)

        return ch

    def note_off(self, midi_note):
        """Stop a note (with fadeout for natural decay)."""
        # Steel pan notes naturally decay, so we let them ring
        # Only stop if explicitly needed
        pass

    def all_notes_off(self):
        """Stop all playing notes."""
        self.pygame.mixer.stop()

    def get_note_range(self):
        """Return the MIDI note range this sampler covers."""
        return MIDI_LOW, MIDI_HIGH


# ---------------------------------------------------------------------------
# MIDI file playback
# ---------------------------------------------------------------------------

def play_midi_file(sampler, midi_path):
    """Play a MIDI file through the steel pan sampler."""
    import mido
    import time

    mid = mido.MidiFile(midi_path)
    print(f"\nPlaying: {midi_path}")
    print(f"  Type: {mid.type}, Tracks: {len(mid.tracks)}, "
          f"Ticks/beat: {mid.ticks_per_beat}")
    print(f"  Duration: {mid.length:.1f}s")
    print(f"  Steel pan range: {midi_to_display_name(MIDI_LOW)} - "
          f"{midi_to_display_name(MIDI_HIGH)}")
    print("\nPress Ctrl+C to stop.\n")

    notes_played = 0
    notes_skipped = 0

    try:
        for msg in mid.play():
            if msg.type == "note_on" and msg.velocity > 0:
                if MIDI_LOW <= msg.note <= MIDI_HIGH:
                    sampler.note_on(msg.note, msg.velocity)
                    name = midi_to_display_name(msg.note)
                    print(f"  {name} (vel={msg.velocity})", end="  ", flush=True)
                    notes_played += 1
                    if notes_played % 8 == 0:
                        print()
                else:
                    notes_skipped += 1
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                sampler.note_off(msg.note)
    except KeyboardInterrupt:
        sampler.all_notes_off()
        print("\n\nStopped.")

    print(f"\n\nFinished: {notes_played} notes played, {notes_skipped} out of range")


# ---------------------------------------------------------------------------
# MIDI file creation
# ---------------------------------------------------------------------------

def create_demo_midi(output_path="steel_pan_demo.mid"):
    """Create a demo MIDI file with a steel pan melody."""
    import mido

    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo (120 BPM)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120)))

    # Set instrument to Steel Drums (MIDI program 114)
    track.append(mido.Message("program_change", program=114, time=0))

    # Track name
    track.append(mido.MetaMessage("track_name", name="Steel Pan Demo"))

    # A Caribbean-flavored melody in C major
    # Each entry: (midi_note, velocity, duration_ticks)
    # Using notes within F3-C6 range
    tpb = 480  # ticks per beat

    melody = [
        # Phrase 1: ascending arpeggio
        (60, 90, tpb),          # C4
        (64, 85, tpb),          # E4
        (67, 95, tpb),          # G4
        (72, 100, tpb * 2),     # C5

        # Phrase 2: descending with rhythm
        (71, 90, tpb // 2),     # B4
        (69, 85, tpb // 2),     # A4
        (67, 90, tpb),          # G4
        (65, 80, tpb // 2),     # F4
        (64, 85, tpb // 2),     # E4
        (62, 90, tpb * 2),      # D4

        # Phrase 3: syncopated rhythm
        (60, 95, tpb // 2),     # C4
        (60, 70, tpb // 4),     # C4 (ghost)
        (64, 90, tpb // 4),     # E4
        (67, 100, tpb),         # G4
        (65, 85, tpb // 2),     # F4
        (64, 90, tpb // 2),     # E4
        (62, 80, tpb),          # D4
        (60, 95, tpb * 2),      # C4

        # Phrase 4: higher register melody
        (72, 90, tpb // 2),     # C5
        (74, 85, tpb // 2),     # D5
        (76, 95, tpb),          # E5
        (79, 100, tpb),         # G5
        (77, 90, tpb // 2),     # F5
        (76, 85, tpb // 2),     # E5
        (74, 90, tpb),          # D5
        (72, 95, tpb * 2),      # C5

        # Ending: chord arpeggios
        (60, 100, tpb // 4),    # C4
        (64, 90, tpb // 4),     # E4
        (67, 95, tpb // 4),     # G4
        (72, 100, tpb // 4),    # C5
        (76, 90, tpb // 4),     # E5
        (79, 95, tpb // 4),     # G5
        (84, 100, tpb * 4),     # C6 (held)
    ]

    for note, vel, dur in melody:
        track.append(mido.Message("note_on", note=note, velocity=vel, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=dur))

    # End of track
    track.append(mido.MetaMessage("end_of_track", time=0))

    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)
    mid.save(output)
    print(f"Created demo MIDI file: {output}")
    return output


# ---------------------------------------------------------------------------
# MIDI controller input
# ---------------------------------------------------------------------------

def list_midi_inputs():
    """List available MIDI input devices."""
    try:
        import mido
        inputs = mido.get_input_names()
        if not inputs:
            print("No MIDI input devices found.")
            print("  Tip: Connect a MIDI keyboard/controller and try again.")
            print("  Tip: Install python-rtmidi for MIDI support: pip install python-rtmidi")
        else:
            print("Available MIDI input devices:")
            for i, name in enumerate(inputs):
                print(f"  [{i}] {name}")
        return inputs
    except Exception as e:
        print(f"Error listing MIDI devices: {e}")
        print("  Install python-rtmidi: pip install python-rtmidi")
        return []


def midi_input_loop(sampler, device_index=None):
    """Listen for MIDI input from a controller and play through sampler."""
    import mido

    inputs = mido.get_input_names()
    if not inputs:
        print("No MIDI input devices found.")
        return

    if device_index is not None and device_index < len(inputs):
        device_name = inputs[device_index]
    else:
        device_name = inputs[0]

    print(f"\nListening on MIDI input: {device_name}")
    print(f"Steel pan range: {midi_to_display_name(MIDI_LOW)} - "
          f"{midi_to_display_name(MIDI_HIGH)}")
    print("Press Ctrl+C to stop.\n")

    try:
        with mido.open_input(device_name) as port:
            for msg in port:
                if msg.type == "note_on" and msg.velocity > 0:
                    sampler.note_on(msg.note, msg.velocity)
                    if MIDI_LOW <= msg.note <= MIDI_HIGH:
                        name = midi_to_display_name(msg.note)
                        print(f"  {name} (vel={msg.velocity})", flush=True)
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    sampler.note_off(msg.note)
    except KeyboardInterrupt:
        sampler.all_notes_off()
        print("\nStopped.")


# ---------------------------------------------------------------------------
# Interactive keyboard mode (pygame)
# ---------------------------------------------------------------------------

def interactive_mode(sampler):
    """Run an interactive keyboard-based steel pan player."""
    import pygame
    import pygame.freetype

    pygame.init()

    WIDTH, HEIGHT = 900, 520
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Steel Pan MIDI Sampler - deepPan")

    # Colors
    BG = (26, 26, 46)
    PANEL = (40, 40, 60)
    KEY_WHITE = (220, 220, 230)
    KEY_BLACK = (50, 50, 70)
    KEY_ACTIVE = (255, 215, 0)
    TEXT = (200, 200, 200)
    DIM = (120, 120, 140)
    HIGHLIGHT = (255, 215, 0)

    font_large = pygame.font.SysFont("Arial", 22, bold=True)
    font_med = pygame.font.SysFont("Arial", 14)
    font_small = pygame.font.SysFont("Arial", 11)

    # Keyboard mapping: computer keys -> MIDI notes
    # Bottom row: C4-B4 (white keys) with sharps on upper row
    # Maps two octaves of piano-style layout
    key_to_midi = {
        # Lower octave (C4-B4) - row Z/S
        pygame.K_z: 60,   # C4
        pygame.K_s: 61,   # C#4
        pygame.K_x: 62,   # D4
        pygame.K_d: 63,   # D#4
        pygame.K_c: 64,   # E4
        pygame.K_v: 65,   # F4
        pygame.K_g: 66,   # F#4
        pygame.K_b: 67,   # G4
        pygame.K_h: 68,   # G#4
        pygame.K_n: 69,   # A4
        pygame.K_j: 70,   # A#4
        pygame.K_m: 71,   # B4
        # Upper octave (C5-C6) - row Q/1
        pygame.K_q: 72,   # C5
        pygame.K_2: 73,   # C#5
        pygame.K_w: 74,   # D5
        pygame.K_3: 75,   # D#5
        pygame.K_e: 76,   # E5
        pygame.K_r: 77,   # F5
        pygame.K_5: 78,   # F#5
        pygame.K_t: 79,   # G5
        pygame.K_6: 80,   # G#5
        pygame.K_y: 81,   # A5
        pygame.K_7: 82,   # A#5
        pygame.K_u: 83,   # B5
        pygame.K_i: 84,   # C6
        # Extra low notes
        pygame.K_COMMA: 53,   # F3
        pygame.K_l: 54,       # F#3
        pygame.K_PERIOD: 55,  # G3
        pygame.K_SEMICOLON: 56,  # G#3
        pygame.K_SLASH: 57,   # A3
        pygame.K_QUOTE: 58,   # A#3
        # A3-B3 via number keys
        pygame.K_9: 57,   # A3 alt
        pygame.K_0: 58,   # A#3 alt
        pygame.K_MINUS: 59, # B3
    }

    active_notes = set()
    velocity = 100

    def draw_piano_key(x, y, w, h, midi_note, is_black=False):
        """Draw a single piano key."""
        active = midi_note in active_notes
        if active:
            color = KEY_ACTIVE
        elif is_black:
            color = KEY_BLACK
        else:
            color = KEY_WHITE

        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, color, rect, border_radius=3)
        if not active:
            pygame.draw.rect(screen, (80, 80, 100), rect, 1, border_radius=3)

        # Note name
        name = midi_to_display_name(midi_note)
        txt_color = (30, 30, 30) if not is_black or active else (180, 180, 180)
        label = font_small.render(name, True, txt_color)
        screen.blit(label, (x + w // 2 - label.get_width() // 2, y + h - 16))

        return rect

    def draw():
        screen.fill(BG)

        # Title
        title = font_large.render("Steel Pan MIDI Sampler", True, HIGHLIGHT)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 15))

        subtitle = font_med.render(
            "urbanPan samples | velocity-layered playback", True, DIM
        )
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 42))

        # Info panel
        pygame.draw.rect(screen, PANEL, (20, 70, WIDTH - 40, 45), border_radius=6)
        info = font_med.render(
            f"Range: {midi_to_display_name(MIDI_LOW)}-{midi_to_display_name(MIDI_HIGH)}  |  "
            f"Velocity: {velocity}  |  "
            f"Samples loaded: {len(sampler.samples)}  |  "
            f"Polyphony: {sampler.max_polyphony}",
            True, TEXT
        )
        screen.blit(info, (35, 82))

        # Draw piano-style keyboard
        # White keys
        white_key_w = 38
        white_key_h = 140
        black_key_w = 24
        black_key_h = 90

        # Two octave sections
        sections = [
            ("F3-B3", 53, 59),
            ("C4-B4", 60, 71),
            ("C5-C6", 72, 84),
        ]

        x_start = 30
        y_keys = 135

        for section_name, low, high in sections:
            # Section label
            lbl = font_small.render(section_name, True, DIM)
            screen.blit(lbl, (x_start, y_keys - 14))

            white_x = x_start
            whites = []
            blacks = []

            for midi_note in range(low, high + 1):
                note_in_octave = midi_note % 12
                is_black = note_in_octave in [1, 3, 6, 8, 10]
                if not is_black:
                    whites.append((white_x, midi_note))
                    white_x += white_key_w + 2

            # Draw white keys first
            for wx, mn in whites:
                draw_piano_key(wx, y_keys, white_key_w, white_key_h, mn, False)

            # Draw black keys on top
            for wx, mn in whites:
                note_in_octave = mn % 12
                # Place black key after this white key if next semitone is black
                next_note = mn + 1
                if next_note <= high and (next_note % 12) in [1, 3, 6, 8, 10]:
                    bx = wx + white_key_w - black_key_w // 2
                    draw_piano_key(bx, y_keys, black_key_w, black_key_h, next_note, True)

            x_start = white_x + 15

        # Keyboard mapping guide
        y_guide = y_keys + white_key_h + 25
        pygame.draw.rect(screen, PANEL, (20, y_guide, WIDTH - 40, 200), border_radius=6)

        guides = [
            "Keyboard Controls:",
            "",
            "Lower octave (C4-B4):  Z X C V B N M = white keys",
            "                       S D   G H J   = black keys",
            "",
            "Upper octave (C5-C6):  Q W E R T Y U I = white keys",
            "                       2 3   5 6 7     = black keys",
            "",
            "Extra low (F3-B3):     , . /  and  L ; '  = chromatic",
            "",
            "UP/DOWN arrows = change velocity  |  ESC = quit",
        ]

        for i, line in enumerate(guides):
            color = HIGHLIGHT if i == 0 else TEXT
            txt = font_small.render(line, True, color)
            screen.blit(txt, (35, y_guide + 8 + i * 17))

        pygame.display.flip()

    # Main loop
    clock = pygame.time.Clock()
    running = True
    print("\nInteractive mode started. Press keys to play steel pan notes.")
    print("Close the window or press ESC to quit.\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    velocity = min(127, velocity + 10)
                elif event.key == pygame.K_DOWN:
                    velocity = max(1, velocity - 10)
                elif event.key in key_to_midi:
                    midi_note = key_to_midi[event.key]
                    sampler.note_on(midi_note, velocity)
                    active_notes.add(midi_note)
                    name = midi_to_display_name(midi_note)
                    print(f"  {name} (vel={velocity})", flush=True)
            elif event.type == pygame.KEYUP:
                if event.key in key_to_midi:
                    midi_note = key_to_midi[event.key]
                    active_notes.discard(midi_note)

        draw()
        clock.tick(30)

    sampler.all_notes_off()
    pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Steel Pan MIDI Sampler - sample-based playback using urbanPan recordings"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download samples from urbanPan GitHub repo"
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download all samples even if they exist"
    )
    parser.add_argument(
        "--play", metavar="FILE",
        help="Play a MIDI file through the steel pan sampler"
    )
    parser.add_argument(
        "--create-demo", action="store_true",
        help="Create a demo MIDI file and play it"
    )
    parser.add_argument(
        "--list-midi", action="store_true",
        help="List available MIDI input devices"
    )
    parser.add_argument(
        "--midi-input", metavar="N", type=int,
        help="Listen for MIDI input from device index N"
    )
    parser.add_argument(
        "--velocity", metavar="V", type=int, default=100,
        help="Default velocity for interactive mode (1-127, default: 100)"
    )
    parser.add_argument(
        "--polyphony", metavar="N", type=int, default=16,
        help="Maximum simultaneous notes (default: 16)"
    )

    args = parser.parse_args()

    # Download-only mode
    if args.download or args.force_download:
        download_samples(force=args.force_download)
        return

    # List MIDI devices
    if args.list_midi:
        list_midi_inputs()
        return

    # Ensure samples are available
    if not ensure_samples():
        print("Error: Could not download samples. Check your internet connection.")
        sys.exit(1)

    # Initialize sampler
    import pygame
    pygame.init()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

    sampler = SteelPanSampler(max_polyphony=args.polyphony)
    sampler.load_samples()

    # MIDI file playback
    if args.play:
        if not os.path.exists(args.play):
            print(f"Error: MIDI file not found: {args.play}")
            sys.exit(1)
        play_midi_file(sampler, args.play)
        return

    # Create and play demo
    if args.create_demo:
        demo_path = create_demo_midi()
        play_midi_file(sampler, demo_path)
        return

    # MIDI controller input
    if args.midi_input is not None:
        midi_input_loop(sampler, args.midi_input)
        return

    # Default: interactive keyboard mode
    interactive_mode(sampler)


if __name__ == "__main__":
    main()
