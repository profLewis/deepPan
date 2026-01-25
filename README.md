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
| I0 | C#6 | [I0_Cs6.wav](sounds/I0_Cs6.wav) |
| I1 | E6 | [I1_E6.wav](sounds/I1_E6.wav) |
| I2 | D6 | [I2_D6.wav](sounds/I2_D6.wav) |
| I3 | C6 | [I3_C6.wav](sounds/I3_C6.wav) |
| I4 | Eb6 | [I4_Eb6.wav](sounds/I4_Eb6.wav) |

### Central Ring (5ths) - 12 notes

| Index | Note | Sound |
|-------|------|-------|
| C0 | F#5 | [C0_Fs5.wav](sounds/C0_Fs5.wav) |
| C1 | B5 | [C1_B5.wav](sounds/C1_B5.wav) |
| C2 | E5 | [C2_E5.wav](sounds/C2_E5.wav) |
| C3 | A5 | [C3_A5.wav](sounds/C3_A5.wav) |
| C4 | D5 | [C4_D5.wav](sounds/C4_D5.wav) |
| C5 | G5 | [C5_G5.wav](sounds/C5_G5.wav) |
| C6 | C5 | [C6_C5.wav](sounds/C6_C5.wav) |
| C7 | F5 | [C7_F5.wav](sounds/C7_F5.wav) |
| C8 | Bb5 | [C8_Bb5.wav](sounds/C8_Bb5.wav) |
| C9 | Eb5 | [C9_Eb5.wav](sounds/C9_Eb5.wav) |
| C10 | Ab5 | [C10_Ab5.wav](sounds/C10_Ab5.wav) |
| C11 | C#5 | [C11_Cs5.wav](sounds/C11_Cs5.wav) |

### Outer Ring (4ths) - 12 notes

| Index | Note | Sound |
|-------|------|-------|
| O0 | F#4 | [O0_Fs4.wav](sounds/O0_Fs4.wav) |
| O1 | B4 | [O1_B4.wav](sounds/O1_B4.wav) |
| O2 | E4 | [O2_E4.wav](sounds/O2_E4.wav) |
| O3 | A4 | [O3_A4.wav](sounds/O3_A4.wav) |
| O4 | D4 | [O4_D4.wav](sounds/O4_D4.wav) |
| O5 | G4 | [O5_G4.wav](sounds/O5_G4.wav) |
| O6 | C4 | [O6_C4.wav](sounds/O6_C4.wav) |
| O7 | F4 | [O7_F4.wav](sounds/O7_F4.wav) |
| O8 | Bb4 | [O8_Bb4.wav](sounds/O8_Bb4.wav) |
| O9 | Eb4 | [O9_Eb4.wav](sounds/O9_Eb4.wav) |
| O10 | Ab4 | [O10_Ab4.wav](sounds/O10_Ab4.wav) |
| O11 | C#4 | [O11_Cs4.wav](sounds/O11_Cs4.wav) |

## 3D Model Object Mapping

### Outer Ring (4ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| O0 | F# | object_58 | object_62 |
| O1 | B | object_57 | object_63 |
| O2 | E | object_56 | object_64 |
| O3 | A | object_55 | object_90 |
| O4 | D | object_54 | object_65 |
| O5 | G | object_53 | object_66 |
| O6 | C | object_59 | object_60 |
| O7 | F | object_52 | object_61 |
| O8 | Bb | object_51 | object_67 |
| O9 | Eb | object_50 | object_88 |
| O10 | Ab | object_49 | object_68 |
| O11 | C# | object_48 | object_69 |

### Central Ring (5ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| C0 | F# | object_25 | object_45 |
| C1 | B | object_24 | object_46 |
| C2 | E | object_23 | object_47 |
| C3 | A | object_22 | object_37 |
| C4 | D | object_21 | object_38 |
| C5 | G | object_20 | object_39 |
| C6 | C | object_73 | object_31 |
| C7 | F | object_30 | object_40 |
| C8 | Bb | object_29 | object_41 |
| C9 | Eb | object_28 | object_42 |
| C10 | Ab | object_27 | object_43 |
| C11 | C# | object_26 | object_44 |

### Inner Ring (6ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| I0 | C# | object_72 | object_36 |
| I1 | E | object_71 | object_32 |
| I2 | D | object_19 | object_33 |
| I3 | C | object_70 | object_34 |
| I4 | Eb | object_18 | object_35 |

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
