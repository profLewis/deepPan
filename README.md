# deepPan

3D model and analysis of a Tenor Steel Pan (Steel Drum).

## Tenor Pan Layout

![Tenor Pan Note Layout](docs/tenor_pan_layout.png)

## Note Pad Structure

The tenor pan consists of **29 note pads** arranged in three concentric rings. Each note pad is composed of:
- **Grove** - the groove boundary that acoustically isolates the note
- **Pan** - the raised playing surface

### Ring Layout

| Ring | Notes | Interval Pattern |
|------|-------|------------------|
| Outer | 12 | 4ths |
| Central | 12 | 5ths |
| Inner | 5 | 6ths |

### Note Mapping

#### Outer Ring (4ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| O0 | F# | object_49 | object_68 |
| O1 | B | object_48 | object_69 |
| O2 | E | object_27 | object_43 |
| O3 | A | object_57 | object_63 |
| O4 | D | object_22 | object_37 |
| O5 | G | object_56 | object_64 |
| O6 | C | object_55 | object_90 |
| O7 | F | object_54 | object_65 |
| O8 | Bb | object_53 | object_66 |
| O9 | Eb | object_52 | object_61 |
| O10 | Ab | object_51 | object_67 |
| O11 | C# | object_50 | object_88 |

#### Central Ring (5ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| C0 | F# | object_28 | object_42 |
| C1 | B | object_26 | object_44 |
| C2 | E | object_18 | object_35 |
| C3 | A | object_25 | object_45 |
| C4 | D | object_71 | object_32 |
| C5 | G | object_24 | object_46 |
| C6 | C | object_23 | object_47 |
| C7 | F | object_21 | object_38 |
| C8 | Bb | object_20 | object_39 |
| C9 | Eb | object_59 | object_60 |
| C10 | Ab | object_30 | object_40 |
| C11 | C# | object_29 | object_41 |

#### Inner Ring (6ths)
| Index | Note | Grove Object | Pan Object |
|-------|------|--------------|------------|
| I0 | C# | object_70 | object_34 |
| I1 | E | object_72 | object_36 |
| I2 | D | object_58 | object_62 |
| I3 | C | object_19 | object_33 |
| I4 | Eb | object_73 | object_31 |

## Files

- `data/Tenor Pan only.obj` - 3D model in Wavefront OBJ format (exported from Rhino)
- `generate_diagram.py` - Python script to analyze the OBJ and generate the layout diagram
- `docs/tenor_pan_layout.png` - Generated diagram of the note layout

## Usage

To regenerate the diagram:

```bash
python3 generate_diagram.py
```

Requires: `matplotlib`, `numpy`
