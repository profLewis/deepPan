# deepPan

3D printable tenor steel pan (steel drum) with modular note pads.

## Project Overview

This project creates **3D printable note pads** for a tenor steel pan. Each of the 29 notes can be individually printed and mounted, allowing for:
- Replacement of damaged notes
- Experimentation with materials and tuning
- Educational exploration of steel pan acoustics
- Custom pan configurations

The note pads are extracted from a 3D scanned tenor pan model and converted into printable solids with integrated mounting hardware.

## How It Works

### 1. Note Pad Extraction
Each note pad is extracted from the original 3D model (`data/Tenor Pan only.obj`) by:
- Identifying the **grove** (groove boundary) that acoustically isolates the note
- Identifying the **pan** (raised playing surface)
- Thickening both surfaces into printable solids (1.5mm thickness)
- Adding a mounting cylinder with internal threads

### 2. Mount System
Each note pad has a **three-part mounting system**:

```
     Note Pad (with external threads)
           │
           ▼
    ┌─────────────┐
    │  Mount Base │  ← Screws onto note pad, has PCB mount
    │  (internal  │     and external threads
    │   threads)  │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Outer Sleeve│  ← Screws onto mount base, protective
    │  (internal  │     housing with grip ridges
    │   threads)  │
    └─────────────┘
```

**Mount Base** (`generate_mount_base.py`):
- Internal threads to grip the note pad
- External threads for the outer sleeve
- Wire notch for routing electronics
- PCB mounting bosses (M2 holes on 16mm grid)
- Dimensions: ~38mm diameter, 12mm height

**Outer Sleeve** (`generate_outer_sleeve.py`):
- Internal threads to grip the mount base
- 12 grip ridges for easy turning
- 12mm wire slit for cable routing
- 10mm access hole in floor for dismantling
- Dimensions: ~48mm diameter, 22mm height

### 3. Thread Design
All threads use a **push-fit ring groove** design (non-helical) for easy 3D printing:
- Pitch: 2.0mm
- Depth: 1.0mm
- Clearance: 0.3mm

## Tenor Pan Layout

![Tenor Pan Note Layout](docs/tenor_pan_layout.png)

The tenor pan has **29 note pads** in three concentric rings:

| Ring | Notes | Interval | Octave |
|------|-------|----------|--------|
| Outer | 12 | 4ths | 4 |
| Central | 12 | 5ths | 5 |
| Inner | 5 | 6ths | 6 |

## Generated Files

### Note Pads (`data/notepads/`)
- `notepad_O0.stl` through `notepad_O11.stl` - Outer ring (12 notes)
- `notepad_C0.stl` through `notepad_C11.stl` - Central ring (12 notes)
- `notepad_I0.stl` through `notepad_I4.stl` - Inner ring (5 notes)
- `notepad_properties.json` - Dimensions and orientations

### Mount Hardware (`data/mounts/`)
- `mount_base.stl` - Mount base (one per note pad)
- `outer_sleeve.stl` - Outer sleeve (one per note pad)

## Usage

### Generate All Note Pads
```bash
python generate_notepad.py --all
```

### Generate Single Note Pad
```bash
python generate_notepad.py O0    # Outer ring, F#4
python generate_notepad.py C5    # Central ring, G5
python generate_notepad.py I2    # Inner ring, D6
```

### Generate Mount Hardware
```bash
python generate_mount_base.py
python generate_outer_sleeve.py
```

### 3D Printing Notes
- All meshes are watertight and manifold
- Recommended layer height: 0.2mm
- Recommended infill: 20-30%
- Support may be needed for note pad overhangs
- Thread clearance designed for FDM printers (0.3mm)

## Interactive Player

Open `index.html` in a browser to play notes by clicking on the pan visualization.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profLewis/deepPan/blob/main/deepPan_Colab.ipynb)

Run the synthesizer in your browser with no installation:
- Interactive sliders for all parameters
- Play notes and sequences
- Download generated WAV files
- Export all 29 notes as ZIP

## Command-Line Player

```bash
# Play a scale
./deepPanPlay "C4 D4 E4 F4 G4 A4 B4 C5"

# With tempo control
./deepPanPlay --bpm 100 "C4 E4 G4 C5"

# List available notes
./deepPanPlay --list
```

## Sound Design Tools

### Interactive Synthesizer (`synth.html`)

A browser-based tool for designing steel pan sounds with real-time parameter control.

**How to run locally:**
```bash
# Option 1: Open directly in browser
open synth.html                           # macOS
xdg-open synth.html                       # Linux
start synth.html                          # Windows

# Option 2: Serve locally (recommended for some browsers)
python -m http.server 8000
# Then open http://localhost:8000/synth.html
```

**Run online (no installation):**
- **GitHub Pages**: Enable Pages on this repo, then access at `https://<username>.github.io/deepPan/synth.html`
- **Direct from GitHub**: Use [raw.githack.com](https://raw.githack.com) to serve directly
- **CodePen/JSFiddle**: Copy the HTML into an online editor for quick testing

**Features:**
- Real-time audio synthesis using Web Audio API
- Sliders for all synthesis parameters (ADSR envelope, harmonics, filter, brightness)
- 5 presets: Default, Bright, Mellow, Bell, Pluck
- Keyboard shortcuts: QWERTY row = Central ring, ASDF row = Outer ring, ZXCV row = Inner ring
- Sequence player for entering melodies
- **CLI command display** shows the equivalent `generate_sounds.py` command

**Parameters:**

| Group | Parameter | Range | Default | Description |
|-------|-----------|-------|---------|-------------|
| Envelope | Attack | 1-200ms | 15ms | Time to reach peak |
| | Decay | 50-2000ms | 500ms | Time to reach sustain |
| | Sustain | 0-100% | 20% | Sustained level |
| | Release | 50-2000ms | 300ms | Time to fade out |
| Harmonics | Fundamental | 0-100% | 100% | Base frequency amplitude |
| | 2nd Harmonic | 0-100% | 30% | Octave above |
| | 3rd Harmonic | 0-100% | 10% | Fifth above octave |
| | 4th Harmonic | 0-100% | 5% | Two octaves above |
| | Sub Bass | 0-100% | 20% | Octave below |
| Character | Detune | 0-20 cents | 2 | Beating effect |
| | Filter | 500-10000Hz | 6000Hz | Low-pass cutoff |
| | Brightness | 0-100% | 50% | High frequency boost |
| Output | Duration | 0.5-3.0s | 1.5s | Note length |
| | Volume | 0-100% | 85% | Output level |

### Sound Generator CLI (`generate_sounds.py`)

Generate WAV files with customizable synthesis parameters.

```bash
# Generate all notes with defaults
python generate_sounds.py

# Use a preset
python generate_sounds.py --preset bright

# Custom parameters
python generate_sounds.py --attack 5 --decay 300 --brightness 80

# Single note
python generate_sounds.py C4

# List all parameters
python generate_sounds.py --list-params

# Available presets
python generate_sounds.py --preset default   # Warm, balanced steel pan
python generate_sounds.py --preset bright    # Crisp, cutting tone
python generate_sounds.py --preset mellow    # Soft, ambient
python generate_sounds.py --preset bell      # Long, ringing
python generate_sounds.py --preset pluck     # Short, percussive
```

The synthesizer and CLI use identical parameters, so you can design sounds in the browser, then regenerate all samples using the CLI command shown.

### Pygame Synthesizer (`synth_pygame.py`)

A native Python application with the same features as the browser version.

**Requirements:**
```bash
pip install pygame numpy
```

**Run:**
```bash
python synth_pygame.py
```

**Controls:**
- **Mouse**: Click note buttons or drag sliders
- **Keyboard**: ASDF row = Outer ring, QWERTY row = Central ring, ZXCV row = Inner ring
- **Presets**: Press 1-5 to load presets
- **ESC**: Quit

**Features:**
- Real-time synthesis (no pre-generated samples)
- Visual ADSR envelope and harmonic spectrum displays
- All the same parameters as synth.html

## File Structure

```
deepPan/
├── data/
│   ├── Tenor Pan only.obj    # Source 3D model
│   ├── mounts/               # Mount hardware STL/OBJ
│   └── notepads/             # Note pad STL/OBJ
├── sounds/                   # Synthesized audio samples
│   └── params.json           # Parameters used for generation
├── docs/                     # Documentation images
├── generate_notepad.py       # Note pad generator
├── generate_mount_base.py    # Mount base generator
├── generate_outer_sleeve.py  # Outer sleeve generator
├── generate_diagram.py       # Layout diagram generator
├── generate_sounds.py        # Sound synthesis CLI
├── generate_interactive.py   # Interactive HTML generator
├── index.html                # Interactive pan player
├── synth.html                # Sound design tool (browser)
├── synth_pygame.py           # Sound design tool (native Python)
├── deepPan_Colab.ipynb       # Google Colab notebook
├── deepPanPlay               # Command-line melody player
└── README.md
```

## Requirements

- Python 3.x
- numpy
- scipy (for audio)
- matplotlib (for diagrams)

## Sound Samples

Each note has a synthesized WAV file in `sounds/`:

### Outer Ring (Octave 4)
F#4, B4, E4, A4, D4, G4, C4, F4, Bb4, Eb4, Ab4, C#4

### Central Ring (Octave 5)
F#5, B5, E5, A5, D5, G5, C5, F5, Bb5, Eb5, Ab5, C#5

### Inner Ring (Octave 6)
C#6, E6, D6, C6, Eb6

## License

This project is for educational and personal use.
