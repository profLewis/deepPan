# Claude Code Session Notes - deepPan

## Session Summary (January 2026)

This session focused on adding **sound design tools** and **interactive features** to the deepPan project.

---

## What Was Built

### 1. Sound Synthesis System

**`generate_sounds.py`** - Complete rewrite with configurable parameters:
- ADSR envelope (attack, decay, sustain, release)
- Harmonics (fundamental, 2nd, 3rd, 4th, sub-bass)
- Character controls (detune, filter, brightness)
- 5 presets: default, bright, mellow, bell, pluck
- CLI arguments for all parameters
- `--params-file` option to load from JSON

**Usage:**
```bash
python generate_sounds.py                          # All notes, defaults
python generate_sounds.py --preset bright          # Use preset
python generate_sounds.py --attack 5 --decay 300   # Custom params
python generate_sounds.py --params-file sound.json # Load from file
```

### 2. Browser Synthesizer (`synth.html`)

Interactive sound design tool with:
- Real-time Web Audio API synthesis
- Sliders for all 14 parameters
- 4 visualization displays:
  - Live audio (waveform/spectrum/lissajous/bars modes)
  - ADSR envelope preview
  - Harmonic spectrum bars
  - Synthesized waveform preview
- ADSR validation (prevents impossible envelopes)
- CLI command display (shows equivalent generate_sounds.py command)
- **Save/Load parameters** to JSON files
- Sequence player with keyboard shortcuts

### 3. Pygame Synthesizer (`synth_pygame.py`)

Native Python desktop application:
- Same parameters as synth.html
- Real-time synthesis (no pre-generated samples)
- Visual ADSR and harmonic displays
- Keyboard shortcuts for notes (QWERTY/ASDF/ZXCV rows)
- Preset loading (keys 1-5)
- **Save/Load parameters** (Ctrl+S/Ctrl+L or buttons)
- Reset button (R key or button)

**Requirements:**
```bash
pip install pygame numpy
```

### 4. Google Colab Notebook (`deepPan_Colab.ipynb`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/profLewis/deepPan/blob/main/deepPan_Colab.ipynb)

Cloud-based synthesizer with:
- ipywidgets sliders for all parameters
- Play notes and sequences
- 5 presets
- Save/load parameters to JSON
- Download single notes as WAV
- Download all 29 notes as ZIP
- Shows equivalent CLI command

### 5. Interactive Player Enhancements (`index.html`)

- **Visualization overlay**: Circular spectrum analyzer inside the pan
  - 4 display modes (Wave, Bars, Spiral, Scope - click to cycle)
  - Off by default (checkbox toggle)
- **Tune file loading**: Load .txt/.tune/.csv files with note sequences

---

## Parameter Reference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| attack | 1-200ms | 15 | Time to peak |
| decay | 50-2000ms | 500 | Time to sustain |
| sustain | 0-100% | 20 | Sustained level |
| release | 50-2000ms | 300 | Fade out time |
| fundamental | 0-100% | 100 | Base frequency |
| harmonic2 | 0-100% | 30 | Octave above |
| harmonic3 | 0-100% | 10 | 5th above octave |
| harmonic4 | 0-100% | 5 | 2 octaves above |
| sub_bass | 0-100% | 20 | Octave below |
| detune | 0-20 cents | 2 | Beating effect |
| filter | 500-10000Hz | 6000 | Low-pass cutoff |
| brightness | 0-100% | 50 | High freq boost |
| duration | 0.5-3.0s | 1.5 | Note length |
| volume | 0-100% | 85 | Output level |

---

## Workflow

### Design → Save → Generate

1. **Design sound** in any tool:
   - `synth.html` (browser)
   - `synth_pygame.py` (desktop)
   - `deepPan_Colab.ipynb` (cloud)

2. **Save parameters** to JSON file

3. **Generate samples**:
   ```bash
   python generate_sounds.py --params-file my_sound.json
   ```

4. **Play** in `index.html` or with `./deepPanPlay`

### JSON Parameter File Format

All tools use the same format:
```json
{
  "attack": 15,
  "decay": 500,
  "sustain": 20,
  "release": 300,
  "fundamental": 100,
  "harmonic2": 30,
  "harmonic3": 10,
  "harmonic4": 5,
  "sub_bass": 20,
  "detune": 2,
  "filter": 6000,
  "brightness": 50,
  "duration": 1.5,
  "volume": 85
}
```

---

## Files Modified/Created This Session

### Created:
- `synth.html` - Browser synthesizer with visualizations
- `synth_pygame.py` - Native Python synthesizer
- `deepPan_Colab.ipynb` - Google Colab notebook
- `CLAUDE_SESSION.md` - This file

### Modified:
- `generate_sounds.py` - Complete rewrite with CLI params and --params-file
- `index.html` - Added visualization overlay and file loading
- `README.md` - Documented all new tools

---

## Git Commits This Session

```
59eef34 Add Colab badge and link to README
b8389fa Add Google Colab notebook
550be74 Add session notes documenting sound design tools
6d5f0f7 Add save/load parameters to synth.html
fca5a9d Add save/load parameters to file
8d7460b Add reset button and update README
470b5c1 Add pygame synthesizer
7a3f923 Add tune file loading
fb90f29 Move visualization inside the pan disk
6a4c287 Add sound design tools and visualizations
```

---

## Tools Summary

| Tool | Platform | Real-time | Save/Load | Visualizations |
|------|----------|-----------|-----------|----------------|
| synth.html | Browser | ✓ | ✓ | 4 displays |
| synth_pygame.py | Desktop | ✓ | ✓ | 2 displays |
| deepPan_Colab.ipynb | Cloud | ✓ | ✓ | - |
| generate_sounds.py | CLI | - | ✓ | - |
| index.html | Browser | ✓* | - | 4 modes |

*Uses pre-generated samples

---

## Future Ideas

- MIDI input support
- More visualization modes
- Recording/export audio from browser
- Multi-note polyphony
- Effects (reverb, delay)
- Touch support for mobile
- Waveform export from Colab

---

## Technical Notes

- All tools share the same parameter format (JSON interchangeable)
- Visualization uses `getBoundingClientRect()` for proper canvas sizing
- ADSR validation ensures attack + decay + release ≤ 95% of duration
- Web Audio uses `MediaElementSourceNode` to route HTML audio through analyser
- Pygame uses `pygame.sndarray.make_sound()` for real-time audio
- Colab uses `IPython.display.Audio` for playback
