#!/usr/bin/env python3
"""
Steel Pan Synthesizer - Pygame Version
Real-time synthesis with interactive sliders.

Usage:
    python synth_pygame.py

Controls:
    - Click and drag sliders to adjust parameters
    - Click note buttons or use keyboard to play
    - Press 1-5 to load presets
    - Press ESC to quit
"""

import pygame
import numpy as np
import sys
import json
from tkinter import filedialog
import tkinter as tk

# Initialize pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Screen setup
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Steel Pan Synthesizer - deepPan")

# Colors
BG_COLOR = (26, 26, 46)
PANEL_COLOR = (40, 40, 60)
SLIDER_BG = (60, 60, 80)
SLIDER_FG = (255, 215, 0)
TEXT_COLOR = (200, 200, 200)
LABEL_COLOR = (150, 150, 150)
BUTTON_OUTER = (192, 57, 43)
BUTTON_CENTRAL = (36, 113, 163)
BUTTON_INNER = (30, 132, 73)
BUTTON_RESET = (100, 100, 100)
BUTTON_SAVE = (46, 134, 193)
BUTTON_LOAD = (39, 174, 96)
WHITE = (255, 255, 255)

# Fonts
font_large = pygame.font.SysFont('Arial', 24, bold=True)
font_medium = pygame.font.SysFont('Arial', 16)
font_small = pygame.font.SysFont('Arial', 12)

# Note frequencies
NOTE_FREQUENCIES = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'Eb': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'Ab': 415.30, 'A': 440.00, 'Bb': 466.16, 'B': 493.88,
}

# Note map (matching the pan layout)
NOTE_MAP = {
    # Outer ring (octave 4)
    'O0': ('F#', 4), 'O1': ('B', 4), 'O2': ('E', 4), 'O3': ('A', 4),
    'O4': ('D', 4), 'O5': ('G', 4), 'O6': ('C', 4), 'O7': ('F', 4),
    'O8': ('Bb', 4), 'O9': ('Eb', 4), 'O10': ('Ab', 4), 'O11': ('C#', 4),
    # Central ring (octave 5)
    'C0': ('F#', 5), 'C1': ('B', 5), 'C2': ('E', 5), 'C3': ('A', 5),
    'C4': ('D', 5), 'C5': ('G', 5), 'C6': ('C', 5), 'C7': ('F', 5),
    'C8': ('Bb', 5), 'C9': ('Eb', 5), 'C10': ('Ab', 5), 'C11': ('C#', 5),
    # Inner ring (octave 6)
    'I0': ('C#', 6), 'I1': ('E', 6), 'I2': ('D', 6), 'I3': ('C', 6), 'I4': ('Eb', 6),
}

# Keyboard mapping
KEY_MAP = {
    pygame.K_a: 'O0', pygame.K_s: 'O1', pygame.K_d: 'O2', pygame.K_f: 'O3',
    pygame.K_g: 'O4', pygame.K_h: 'O5', pygame.K_j: 'O6', pygame.K_k: 'O7',
    pygame.K_l: 'O8',
    pygame.K_q: 'C0', pygame.K_w: 'C1', pygame.K_e: 'C2', pygame.K_r: 'C3',
    pygame.K_t: 'C4', pygame.K_y: 'C5', pygame.K_u: 'C6', pygame.K_i: 'C7',
    pygame.K_o: 'C8', pygame.K_p: 'C9',
    pygame.K_z: 'I0', pygame.K_x: 'I1', pygame.K_c: 'I2', pygame.K_v: 'I3', pygame.K_b: 'I4',
}

# Default parameters
params = {
    'attack': 15,
    'decay': 500,
    'sustain': 20,
    'release': 300,
    'fundamental': 100,
    'harmonic2': 30,
    'harmonic3': 10,
    'harmonic4': 5,
    'sub_bass': 20,
    'detune': 2,
    'filter': 6000,
    'brightness': 50,
    'duration': 1.5,
    'volume': 85,
}

# Presets
PRESETS = {
    'default': dict(attack=15, decay=500, sustain=20, release=300,
                    fundamental=100, harmonic2=30, harmonic3=10, harmonic4=5,
                    sub_bass=20, detune=2, filter=6000, brightness=50, duration=1.5, volume=85),
    'bright': dict(attack=5, decay=300, sustain=10, release=200,
                   fundamental=80, harmonic2=50, harmonic3=30, harmonic4=20,
                   sub_bass=10, detune=3, filter=10000, brightness=80, duration=1.5, volume=85),
    'mellow': dict(attack=30, decay=800, sustain=30, release=500,
                   fundamental=100, harmonic2=15, harmonic3=5, harmonic4=2,
                   sub_bass=30, detune=1, filter=3000, brightness=30, duration=1.5, volume=85),
    'bell': dict(attack=2, decay=1500, sustain=5, release=1000,
                 fundamental=70, harmonic2=60, harmonic3=40, harmonic4=30,
                 sub_bass=5, detune=5, filter=8000, brightness=60, duration=1.5, volume=85),
    'pluck': dict(attack=1, decay=200, sustain=0, release=100,
                  fundamental=100, harmonic2=40, harmonic3=20, harmonic4=10,
                  sub_bass=15, detune=0, filter=5000, brightness=50, duration=1.5, volume=85),
}

# Slider definitions
SLIDERS = [
    # (name, key, min, max, x, y, unit)
    ('Attack', 'attack', 1, 200, 620, 80, 'ms'),
    ('Decay', 'decay', 50, 2000, 620, 120, 'ms'),
    ('Sustain', 'sustain', 0, 100, 620, 160, '%'),
    ('Release', 'release', 50, 2000, 620, 200, 'ms'),
    ('Fundamental', 'fundamental', 0, 100, 620, 260, '%'),
    ('Harmonic 2', 'harmonic2', 0, 100, 620, 300, '%'),
    ('Harmonic 3', 'harmonic3', 0, 100, 620, 340, '%'),
    ('Harmonic 4', 'harmonic4', 0, 100, 620, 380, '%'),
    ('Sub Bass', 'sub_bass', 0, 100, 620, 420, '%'),
    ('Detune', 'detune', 0, 20, 620, 480, 'ct'),
    ('Filter', 'filter', 500, 10000, 620, 520, 'Hz'),
    ('Brightness', 'brightness', 0, 100, 620, 560, '%'),
    ('Volume', 'volume', 0, 100, 620, 620, '%'),
]

SLIDER_WIDTH = 150
SLIDER_HEIGHT = 8


class Slider:
    def __init__(self, name, key, min_val, max_val, x, y, unit):
        self.name = name
        self.key = key
        self.min_val = min_val
        self.max_val = max_val
        self.x = x
        self.y = y
        self.unit = unit
        self.dragging = False

    def draw(self, surface):
        # Label
        label = font_small.render(self.name, True, LABEL_COLOR)
        surface.blit(label, (self.x, self.y - 18))

        # Value
        val = params[self.key]
        if self.key == 'duration':
            val_text = f"{val:.1f}{self.unit}"
        else:
            val_text = f"{int(val)}{self.unit}"
        val_render = font_small.render(val_text, True, TEXT_COLOR)
        surface.blit(val_render, (self.x + SLIDER_WIDTH + 10, self.y - 5))

        # Background track
        pygame.draw.rect(surface, SLIDER_BG, (self.x, self.y, SLIDER_WIDTH, SLIDER_HEIGHT), border_radius=4)

        # Filled portion
        ratio = (params[self.key] - self.min_val) / (self.max_val - self.min_val)
        fill_width = int(ratio * SLIDER_WIDTH)
        pygame.draw.rect(surface, SLIDER_FG, (self.x, self.y, fill_width, SLIDER_HEIGHT), border_radius=4)

        # Handle
        handle_x = self.x + fill_width
        pygame.draw.circle(surface, WHITE, (handle_x, self.y + SLIDER_HEIGHT // 2), 8)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if self.x - 10 <= mx <= self.x + SLIDER_WIDTH + 10 and self.y - 10 <= my <= self.y + 20:
                self.dragging = True
                self._update_value(mx)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])

    def _update_value(self, mx):
        ratio = max(0, min(1, (mx - self.x) / SLIDER_WIDTH))
        val = self.min_val + ratio * (self.max_val - self.min_val)
        if self.key == 'duration':
            params[self.key] = round(val, 1)
        else:
            params[self.key] = int(val)


class NoteButton:
    def __init__(self, idx, note, octave, x, y, ring):
        self.idx = idx
        self.note = note
        self.octave = octave
        self.x = x
        self.y = y
        self.ring = ring
        self.width = 45
        self.height = 35
        self.playing = False
        self.play_time = 0

    def draw(self, surface):
        color = {'outer': BUTTON_OUTER, 'central': BUTTON_CENTRAL, 'inner': BUTTON_INNER}[self.ring]

        # Brighten if playing
        if self.playing and pygame.time.get_ticks() - self.play_time < 200:
            color = tuple(min(255, c + 80) for c in color)

        pygame.draw.rect(surface, color, (self.x, self.y, self.width, self.height), border_radius=5)

        # Note name
        text = font_medium.render(f"{self.note}{self.octave}", True, WHITE)
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        surface.blit(text, text_rect)

    def contains(self, pos):
        return self.x <= pos[0] <= self.x + self.width and self.y <= pos[1] <= self.y + self.height

    def play(self):
        self.playing = True
        self.play_time = pygame.time.get_ticks()


def get_frequency(note_name, octave):
    """Get frequency for a note in a given octave."""
    base_freq = NOTE_FREQUENCIES.get(note_name)
    if base_freq is None:
        return 440
    return base_freq * (2 ** (octave - 4))


def generate_note(frequency):
    """Generate a steel pan note with current parameters."""
    sample_rate = 44100
    duration = params['duration']
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Convert parameters
    attack_time = params['attack'] / 1000
    decay_time = params['decay'] / 1000
    sustain_level = params['sustain'] / 100
    release_time = params['release'] / 1000
    detune_cents = params['detune']

    # Partials
    partials = [
        (0.50, params['sub_bass'] / 100, 1.0),
        (1.00 - detune_cents/1000, params['fundamental'] / 100 * 0.1, 1.0),
        (1.00, params['fundamental'] / 100, 1.0),
        (1.00 + detune_cents/1000, params['fundamental'] / 100 * 0.1, 1.0),
        (2.00, params['harmonic2'] / 100, 1.3),
        (3.00, params['harmonic3'] / 100, 1.8),
        (4.00, params['harmonic4'] / 100, 2.2),
    ]

    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)

    sound = np.zeros_like(t)

    for ratio, amp, decay_mult in partials:
        if amp < 0.01:
            continue

        freq = frequency * ratio
        if freq > sample_rate / 2 - 500 or freq < 30:
            continue

        # Envelope
        env = np.ones_like(t)

        if attack_samples > 0:
            attack = np.sin(np.linspace(0, np.pi/2, attack_samples)) ** 1.5
            env[:attack_samples] = attack

        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, len(t))
        if decay_end > decay_start:
            decay = np.linspace(1.0, sustain_level, decay_end - decay_start)
            env[decay_start:decay_end] = decay

        sustain_start = decay_end
        sustain_tau = decay_time / decay_mult
        if sustain_start < len(t):
            env[sustain_start:] = sustain_level * np.exp(
                -(t[sustain_start:] - t[sustain_start]) / sustain_tau
            )

        phase = np.random.uniform(0, 2 * np.pi)
        partial_sound = amp * env * np.sin(2 * np.pi * freq * t + phase)
        sound += partial_sound

    # Simple low-pass approximation using moving average
    filter_cutoff = params['filter']
    if filter_cutoff < 8000:
        window_size = max(1, int(sample_rate / filter_cutoff / 2))
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            sound = np.convolve(sound, kernel, mode='same')

    # Soft limiting
    sound = np.tanh(sound * 1.2) / 1.2

    # Normalize and apply volume
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound = sound / max_val * (params['volume'] / 100)

    # Convert to 16-bit stereo
    sound_int = (sound * 32767).astype(np.int16)
    stereo = np.column_stack((sound_int, sound_int))

    return stereo


def play_note(note_name, octave):
    """Generate and play a note."""
    freq = get_frequency(note_name, octave)
    samples = generate_note(freq)

    # Create pygame sound
    sound = pygame.sndarray.make_sound(samples)
    sound.play()


def draw_envelope_preview(surface, x, y, width, height):
    """Draw ADSR envelope preview."""
    pygame.draw.rect(surface, PANEL_COLOR, (x, y, width, height), border_radius=5)

    # Title
    title = font_small.render("ADSR Envelope", True, LABEL_COLOR)
    surface.blit(title, (x + 5, y + 5))

    # Calculate envelope points
    attack = params['attack']
    decay = params['decay']
    sustain = params['sustain'] / 100
    release = params['release']
    duration = params['duration'] * 1000

    total = min(attack + decay + release + 200, duration)
    scale = (width - 20) / total

    pad = 25
    h = height - 35

    points = [
        (x + 10, y + pad + h),  # Start
        (x + 10 + attack * scale, y + pad),  # Attack peak
        (x + 10 + (attack + decay) * scale, y + pad + h * (1 - sustain)),  # Decay end
        (x + 10 + (duration - release) * scale, y + pad + h * (1 - sustain)),  # Sustain end
        (x + 10 + duration * scale, y + pad + h),  # Release end
    ]

    # Clamp points to box
    points = [(min(max(px, x + 10), x + width - 10), py) for px, py in points]

    pygame.draw.lines(surface, SLIDER_FG, False, points, 2)


def draw_harmonic_preview(surface, x, y, width, height):
    """Draw harmonic spectrum preview."""
    pygame.draw.rect(surface, PANEL_COLOR, (x, y, width, height), border_radius=5)

    title = font_small.render("Harmonics", True, LABEL_COLOR)
    surface.blit(title, (x + 5, y + 5))

    harmonics = [
        ('Sub', params['sub_bass'], (136, 68, 255)),
        ('Fund', params['fundamental'], (255, 215, 0)),
        ('2nd', params['harmonic2'], (0, 170, 255)),
        ('3rd', params['harmonic3'], (0, 255, 136)),
        ('4th', params['harmonic4'], (255, 102, 68)),
    ]

    bar_width = (width - 20) // len(harmonics)
    max_height = height - 45

    for i, (name, value, color) in enumerate(harmonics):
        bar_x = x + 10 + i * bar_width
        bar_height = int(value / 100 * max_height)
        bar_y = y + height - 10 - bar_height

        pygame.draw.rect(surface, color, (bar_x + 2, bar_y, bar_width - 4, bar_height))

        # Label
        label = font_small.render(name, True, LABEL_COLOR)
        label_rect = label.get_rect(center=(bar_x + bar_width // 2, y + height - 5))
        surface.blit(label, label_rect)


def save_params_to_file():
    """Save current parameters to a JSON file."""
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()
    
    filepath = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Save Parameters"
    )
    root.destroy()
    
    if filepath:
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        return filepath
    return None


def load_params_from_file():
    """Load parameters from a JSON file."""
    root = tk.Tk()
    root.withdraw()
    
    filepath = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="Load Parameters"
    )
    root.destroy()
    
    if filepath:
        try:
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            # Update only valid keys
            for key in params:
                if key in loaded:
                    params[key] = loaded[key]
            return filepath
        except Exception as e:
            print(f"Error loading file: {e}")
    return None


def main():
    clock = pygame.time.Clock()

    # Create sliders
    sliders = [Slider(*s) for s in SLIDERS]

    # Create note buttons
    buttons = []

    # Outer ring
    for i in range(12):
        idx = f'O{i}'
        note, octave = NOTE_MAP[idx]
        x = 30 + (i % 6) * 50
        y = 80 + (i // 6) * 40
        buttons.append(NoteButton(idx, note, octave, x, y, 'outer'))

    # Central ring
    for i in range(12):
        idx = f'C{i}'
        note, octave = NOTE_MAP[idx]
        x = 30 + (i % 6) * 50
        y = 180 + (i // 6) * 40
        buttons.append(NoteButton(idx, note, octave, x, y, 'central'))

    # Inner ring
    for i in range(5):
        idx = f'I{i}'
        note, octave = NOTE_MAP[idx]
        x = 30 + i * 50
        y = 280
        buttons.append(NoteButton(idx, note, octave, x, y, 'inner'))

    current_note = None
    status_msg = ""
    status_time = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_MAP:
                    idx = KEY_MAP[event.key]
                    note, octave = NOTE_MAP[idx]
                    play_note(note, octave)
                    current_note = f"{note}{octave}"
                    for btn in buttons:
                        if btn.idx == idx:
                            btn.play()
                elif event.key == pygame.K_1:
                    params.update(PRESETS['default'])
                elif event.key == pygame.K_2:
                    params.update(PRESETS['bright'])
                elif event.key == pygame.K_3:
                    params.update(PRESETS['mellow'])
                elif event.key == pygame.K_4:
                    params.update(PRESETS['bell'])
                elif event.key == pygame.K_5:
                    params.update(PRESETS['pluck'])
                elif event.key == pygame.K_0 or event.key == pygame.K_r:
                    params.update(PRESETS['default'])
                elif event.key == pygame.K_s and (event.mod & pygame.KMOD_CTRL or event.mod & pygame.KMOD_META):
                    saved = save_params_to_file()
                    if saved:
                        status_msg = f"Saved: {saved.split('/')[-1]}"
                        status_time = pygame.time.get_ticks()
                elif event.key == pygame.K_l and (event.mod & pygame.KMOD_CTRL or event.mod & pygame.KMOD_META):
                    loaded = load_params_from_file()
                    if loaded:
                        status_msg = f"Loaded: {loaded.split('/')[-1]}"
                        status_time = pygame.time.get_ticks()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check reset/save/load buttons
                reset_rect = pygame.Rect(30, 560, 70, 30)
                save_rect = pygame.Rect(110, 560, 70, 30)
                load_rect = pygame.Rect(190, 560, 70, 30)
                
                if reset_rect.collidepoint(event.pos):
                    params.update(PRESETS['default'])
                    status_msg = "Reset to defaults"
                    status_time = pygame.time.get_ticks()
                elif save_rect.collidepoint(event.pos):
                    saved = save_params_to_file()
                    if saved:
                        status_msg = f"Saved: {saved.split('/')[-1]}"
                        status_time = pygame.time.get_ticks()
                elif load_rect.collidepoint(event.pos):
                    loaded = load_params_from_file()
                    if loaded:
                        status_msg = f"Loaded: {loaded.split('/')[-1]}"
                        status_time = pygame.time.get_ticks()
                
                for btn in buttons:
                    if btn.contains(event.pos):
                        play_note(btn.note, btn.octave)
                        current_note = f"{btn.note}{btn.octave}"
                        btn.play()

            for slider in sliders:
                slider.handle_event(event)

        # Draw
        screen.fill(BG_COLOR)

        # Title
        title = font_large.render("Steel Pan Synthesizer", True, SLIDER_FG)
        screen.blit(title, (20, 20))

        subtitle = font_small.render("deepPan - Pygame Edition", True, LABEL_COLOR)
        screen.blit(subtitle, (20, 50))

        # Ring labels
        outer_label = font_small.render("Outer Ring (Octave 4) - Keys: A-L", True, BUTTON_OUTER)
        screen.blit(outer_label, (30, 62))

        central_label = font_small.render("Central Ring (Octave 5) - Keys: Q-P", True, BUTTON_CENTRAL)
        screen.blit(central_label, (30, 162))

        inner_label = font_small.render("Inner Ring (Octave 6) - Keys: Z-B", True, BUTTON_INNER)
        screen.blit(inner_label, (30, 262))

        # Draw buttons
        for btn in buttons:
            btn.draw(screen)

        # Current note display
        if current_note:
            note_display = font_large.render(f"Playing: {current_note}", True, WHITE)
            screen.blit(note_display, (30, 340))

        # Preset hints
        preset_text = font_small.render("Presets: 1=Default  2=Bright  3=Mellow  4=Bell  5=Pluck", True, LABEL_COLOR)
        screen.blit(preset_text, (30, 380))

        # Draw previews
        draw_envelope_preview(screen, 30, 420, 250, 120)
        draw_harmonic_preview(screen, 300, 420, 280, 120)

        # Sliders panel
        pygame.draw.rect(screen, PANEL_COLOR, (600, 50, 280, 620), border_radius=10)
        panel_title = font_medium.render("Parameters", True, TEXT_COLOR)
        screen.blit(panel_title, (620, 55))

        for slider in sliders:
            slider.draw(screen)

        # Reset/Save/Load buttons
        reset_rect = pygame.Rect(30, 560, 70, 30)
        save_rect = pygame.Rect(110, 560, 70, 30)
        load_rect = pygame.Rect(190, 560, 70, 30)
        
        pygame.draw.rect(screen, BUTTON_RESET, reset_rect, border_radius=5)
        reset_text = font_medium.render("Reset", True, WHITE)
        screen.blit(reset_text, reset_text.get_rect(center=reset_rect.center))
        
        pygame.draw.rect(screen, BUTTON_SAVE, save_rect, border_radius=5)
        save_text = font_medium.render("Save", True, WHITE)
        screen.blit(save_text, save_text.get_rect(center=save_rect.center))
        
        pygame.draw.rect(screen, BUTTON_LOAD, load_rect, border_radius=5)
        load_text = font_medium.render("Load", True, WHITE)
        screen.blit(load_text, load_text.get_rect(center=load_rect.center))
        
        # Status message (show for 3 seconds)
        if status_msg and pygame.time.get_ticks() - status_time < 3000:
            status_render = font_small.render(status_msg, True, SLIDER_FG)
            screen.blit(status_render, (30, 600))

        # Instructions
        inst = font_small.render("Keyboard: notes | R=Reset | Ctrl+S=Save | Ctrl+L=Load | ESC=Quit", True, LABEL_COLOR)
        screen.blit(inst, (30, HEIGHT - 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
