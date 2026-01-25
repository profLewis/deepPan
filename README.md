# deepPan

3D model and analysis of a Tenor Steel Pan (Steel Drum).

## Tenor Pan Layout

![Tenor Pan Note Layout](docs/tenor_pan_layout.png)

*Top-down view projected from the 3D model geometry.*

## Note Pad Structure

The tenor pan consists of **29 note pads** arranged in three concentric rings. Each note pad is composed of:
- **Grove** - the groove boundary that acoustically isolates the note
- **Pan** - the raised playing surface

### Ring Layout

| Ring | Notes | Interval Pattern | Octave |
|------|-------|------------------|--------|
| Outer | 12 | 4ths | 4 |
| Central | 12 | 5ths | 5 |
| Inner | 5 | 6ths | 6 |

## Sound Samples

Click to play each note:

### Inner Ring (6ths) - 5 notes

| Index | Note | Sound |
|-------|------|-------|
| I0 | Eb6 | [I0_Eb6.wav](sounds/I0_Eb6.wav) |
| I1 | C6 | [I1_C6.wav](sounds/I1_C6.wav) |
| I2 | E6 | [I2_E6.wav](sounds/I2_E6.wav) |
| I3 | D6 | [I3_D6.wav](sounds/I3_D6.wav) |
| I4 | C#6 | [I4_Cs6.wav](sounds/I4_Cs6.wav) |

### Central Ring (5ths) - 12 notes

| Index | Note | Sound |
|-------|------|-------|
| C0 | C#5 | [C0_Cs5.wav](sounds/C0_Cs5.wav) |
| C1 | Ab5 | [C1_Ab5.wav](sounds/C1_Ab5.wav) |
| C2 | Eb5 | [C2_Eb5.wav](sounds/C2_Eb5.wav) |
| C3 | Bb5 | [C3_Bb5.wav](sounds/C3_Bb5.wav) |
| C4 | F5 | [C4_F5.wav](sounds/C4_F5.wav) |
| C5 | C5 | [C5_C5.wav](sounds/C5_C5.wav) |
| C6 | G5 | [C6_G5.wav](sounds/C6_G5.wav) |
| C7 | D5 | [C7_D5.wav](sounds/C7_D5.wav) |
| C8 | A5 | [C8_A5.wav](sounds/C8_A5.wav) |
| C9 | E5 | [C9_E5.wav](sounds/C9_E5.wav) |
| C10 | B5 | [C10_B5.wav](sounds/C10_B5.wav) |
| C11 | F#5 | [C11_Fs5.wav](sounds/C11_Fs5.wav) |

### Outer Ring (4ths) - 12 notes

| Index | Note | Sound |
|-------|------|-------|
| O0 | Bb4 | [O0_Bb4.wav](sounds/O0_Bb4.wav) |
| O1 | F4 | [O1_F4.wav](sounds/O1_F4.wav) |
| O2 | C4 | [O2_C4.wav](sounds/O2_C4.wav) |
| O3 | G4 | [O3_G4.wav](sounds/O3_G4.wav) |
| O4 | D4 | [O4_D4.wav](sounds/O4_D4.wav) |
| O5 | A4 | [O5_A4.wav](sounds/O5_A4.wav) |
| O6 | E4 | [O6_E4.wav](sounds/O6_E4.wav) |
| O7 | B4 | [O7_B4.wav](sounds/O7_B4.wav) |
| O8 | F#4 | [O8_Fs4.wav](sounds/O8_Fs4.wav) |
| O9 | C#4 | [O9_Cs4.wav](sounds/O9_Cs4.wav) |
| O10 | Ab4 | [O10_Ab4.wav](sounds/O10_Ab4.wav) |
| O11 | Eb4 | [O11_Eb4.wav](sounds/O11_Eb4.wav) |

## 3D Model Object Mapping

### Outer Ring (4ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| O0 | Bb | object_58 | object_62 |
| O1 | F | object_57 | object_63 |
| O2 | C | object_56 | object_64 |
| O3 | G | object_55 | object_90 |
| O4 | D | object_54 | object_65 |
| O5 | A | object_53 | object_66 |
| O6 | E | object_59 | object_60 |
| O7 | B | object_52 | object_61 |
| O8 | F# | object_51 | object_67 |
| O9 | C# | object_50 | object_88 |
| O10 | Ab | object_49 | object_68 |
| O11 | Eb | object_48 | object_69 |

### Central Ring (5ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| C0 | C# | object_25 | object_45 |
| C1 | Ab | object_24 | object_46 |
| C2 | Eb | object_23 | object_47 |
| C3 | Bb | object_22 | object_37 |
| C4 | F | object_21 | object_38 |
| C5 | C | object_20 | object_39 |
| C6 | G | object_73 | object_31 |
| C7 | D | object_30 | object_40 |
| C8 | A | object_29 | object_41 |
| C9 | E | object_28 | object_42 |
| C10 | B | object_27 | object_43 |
| C11 | F# | object_26 | object_44 |

### Inner Ring (6ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| I0 | Eb | object_72 | object_36 |
| I1 | C | object_71 | object_32 |
| I2 | E | object_19 | object_33 |
| I3 | D | object_70 | object_34 |
| I4 | C# | object_18 | object_35 |

## Files

- `data/Tenor Pan only.obj` - 3D model in Wavefront OBJ format (exported from Rhino)
- `generate_diagram.py` - Python script to analyze the OBJ and generate the layout diagram
- `generate_sounds.py` - Python script to synthesize steel pan sounds for each note
- `docs/tenor_pan_layout.png` - Generated diagram of the note layout (from 3D geometry)
- `sounds/` - Synthesized WAV files for each note

## Usage

Generate the layout diagram:
```bash
python3 generate_diagram.py
```

Generate sound files:
```bash
python3 generate_sounds.py
```

### Requirements

- Python 3.x
- numpy
- scipy
- matplotlib
