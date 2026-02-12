---
layout: default
---

## Interactive Tools

<div style="display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0;">
  <a href="player.html" style="display: inline-block; padding: 12px 24px; background: #157878; color: white; text-decoration: none; border-radius: 6px; font-weight: bold;">Play the Pan</a>
  <a href="synth.html" style="display: inline-block; padding: 12px 24px; background: #157878; color: white; text-decoration: none; border-radius: 6px; font-weight: bold;">Sound Designer</a>
  <a href="https://colab.research.google.com/github/profLewis/deepPan/blob/main/deepPan_Colab.ipynb" style="display: inline-block;"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

- **[Interactive Pan Player](player.html)** -- Click notes on a steel pan to hear them. Includes sequence playback and 4 visualization modes.
- **[Sound Designer](synth.html)** -- Design steel pan sounds with real-time synthesis, 14 parameter controls, and 5 presets.
- **[Google Colab Notebook](https://colab.research.google.com/github/profLewis/deepPan/blob/main/deepPan_Colab.ipynb)** -- Run the synthesizer in the cloud with no installation.

## Project Overview

This project creates a **3D printable tenor steel pan** from a 3D scanned model. The pan is split into printable sections with individually mountable note pads, enabling replacement of damaged notes, experimentation with materials and tuning, and educational exploration of steel pan acoustics.

## Tenor Pan Layout

![Tenor Pan Note Layout](docs/tenor_pan_layout.png)

The tenor pan has **29 note pads** in three concentric rings:

| Ring | Notes | Interval | Octave |
|------|-------|----------|--------|
| Outer | 12 | 4ths | 4 |
| Central | 12 | 5ths | 5 |
| Inner | 5 | 6ths | 6 |

## Section Generator

The pan's outer ring is split into **6 printable sections** of 60° each, designed to fit a Bambu Lab P1S printer (256 x 256 x 256 mm). Each section contains 2 outer-ring notes and assembles together to reconstruct the full pan shell.

| Section | Angle | Notes |
|---------|-------|-------|
| S0 | 15° -- 75° | O2, O1 |
| S1 | 75° -- 135° | O0, O11 |
| S2 | 135° -- 195° | O10, O9 |
| S3 | 195° -- 255° | O8, O7 |
| S4 | 255° -- 315° | O6, O5 |
| S5 | 315° -- 15° | O4, O3 |

Each section includes:
- **Notepad pockets** -- Sunken recesses (1.5mm deep) matching each note's shape, with M4 boss through-holes for mounting
- **Profile-following support walls** -- Radial walls along each cut edge whose top profile follows the drum surface, with M4 bolt holes and finger joints for alignment
- **Inner arc support** -- Curved wall at R=145mm with studs and cable routing hole
- **Drum wall** -- Original scanned drum geometry preserved without subdivision

```bash
# Generate all 6 sections
python generate_quarter.py --all

# Generate a single section
python generate_quarter.py S0

# View in Blender
blender --python view_all_sections.py
```

## Note Pads

Each of the 29 note pads is extracted from the 3D scan and converted into a printable solid with integrated mounting hardware:

1. **Note Pad** -- Thickened to 1.5mm with a threaded mounting cylinder and 4 corner bosses
2. **Mount Base** -- Screws onto the note pad, includes PCB mounting bosses (M2 holes) for electronics
3. **Outer Sleeve** -- Protective housing with 12 grip ridges and cable routing

All threads use a push-fit ring groove design (2mm pitch, 1mm depth, 0.3mm clearance) optimized for FDM printing.

```bash
# Generate all 29 note pads
python generate_notepad.py --all

# Generate a specific note pad
python generate_notepad.py O5
```

## Sound Design

The synthesizer generates realistic steel pan tones using additive synthesis with 14 adjustable parameters:

| Group | Parameters |
|-------|-----------|
| **Envelope** | Attack, Decay, Sustain, Release |
| **Harmonics** | Fundamental, 2nd/3rd/4th Harmonic, Sub Bass |
| **Character** | Detune, Filter Cutoff, Brightness |
| **Output** | Duration, Volume |

Five built-in presets: Default, Bright, Mellow, Bell, and Pluck.

## Command-Line Tools

```bash
# Generate all sound samples
python generate_sounds.py

# Use a preset
python generate_sounds.py --preset bright

# Play a melody
./deepPanPlay "C4 E4 G4 C5"

# Analyze a WAV file to extract parameters
python analyze_audio.py sample.wav --output params.json
```

## Pipeline Summary

| Step | Script | Output |
|------|--------|--------|
| Extract pan surface | `extract_pan_surface.py` | Centered bowl mesh |
| Generate note pads | `generate_notepad.py --all` | 29 note pad OBJ/STL + properties |
| Generate sections | `generate_quarter.py --all` | 6 section OBJ/STL + properties |

## Source

View the full source and documentation on [GitHub](https://github.com/profLewis/deepPan).
