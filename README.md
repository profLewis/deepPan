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

## Command-Line Player

```bash
# Play a scale
./deepPanPlay "C4 D4 E4 F4 G4 A4 B4 C5"

# With tempo control
./deepPanPlay --bpm 100 "C4 E4 G4 C5"

# List available notes
./deepPanPlay --list
```

## File Structure

```
deepPan/
├── data/
│   ├── Tenor Pan only.obj    # Source 3D model
│   ├── mounts/               # Mount hardware STL/OBJ
│   └── notepads/             # Note pad STL/OBJ
├── sounds/                   # Synthesized audio samples
├── docs/                     # Documentation images
├── generate_notepad.py       # Note pad generator
├── generate_mount_base.py    # Mount base generator
├── generate_outer_sleeve.py  # Outer sleeve generator
├── generate_diagram.py       # Layout diagram generator
├── generate_sounds.py        # Audio synthesis
├── generate_interactive.py   # Interactive HTML generator
├── index.html                # Interactive player
├── deepPanPlay               # Command-line player
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
