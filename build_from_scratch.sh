#!/usr/bin/env bash
# Build pipeline_output/pan_printable.{obj,stl} from scratch.
#
# Inputs:
#   data/Tenor Pan only.obj
#   data/pan_centroid_offset.json
#   data/notepads/notepad_properties.json
#   (all Python scripts, generate_sector.py NOTE_MAPPING)
#
# Outputs (all under pipeline_output/):
#   pan_printable.{obj,stl}            — final fused, watertight, printable model
#   pan_holes_solid_merged.{obj,stl}   — body with central+inner groove ellipsoids merged
#   groove_ellipsoids_{central,inner,outer}.{obj,stl}
#   pan_piece_central.stl, pan_piece_outer_0..5.stl  — 7 split parts
#
# Run end-to-end (≈ 40–60 min total):
#   ./build_from_scratch.sh

set -euo pipefail

BLENDER=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
SOURCE="data/Tenor Pan only.obj"

step() { echo; echo "=========================================="; echo " $* "; echo "=========================================="; }

[ -f "$SOURCE" ] || { echo "FATAL: missing source $SOURCE" >&2; exit 1; }
[ -x "$BLENDER" ] || { echo "FATAL: Blender not at $BLENDER (set \$BLENDER to override)" >&2; exit 1; }

mkdir -p data/quarters data/grooves data/notepads pipeline_output

step "1/9  Extract groove surfaces from source bowl"
python3 generate_grooves.py

step "2/9  Solidify grooves (-Y extrusion, 5mm)"
python3 solidify_grooves.py

step "3/9  Generate 29 notepad meshes (no screw holes — pad faces stay solid)"
python3 generate_notepad.py --all --no-screw-holes

step "4/9  Build pan body (groove faces kept in body)"
python3 generate_pan_holes.py

step "5/9  Extend outer-ring pad-hole boundaries (1mm tangent, 4-ring falloff)"
python3 extend_pan_holes.py --extension=1.0 --skip-inner --r-boundary=150 --rings=4 --iters=60

step "6/9  Thicken body to 5mm solid (offset=+1)"
"$BLENDER" --background --python thicken_blender.py -- \
    data/quarters/pan_holes.obj --offset=1.0

step "7/9  Generate per-pad groove replacement ellipsoids (SVD-oriented, 15% XZ margin)"
python3 make_groove_ellipsoids.py

step "8/9  Merge body with central+inner ellipsoids (EXACT boolean)"
"$BLENDER" --background --python merge_body_with_ellipsoids.py -- \
    --rings=central,inner --solver=EXACT

step "9/9  Build printable: + pads + outer grooves → voxel remesh → strip debris"
"$BLENDER" --background --python make_printable.py -- --voxel=0.5
python3 strip_debris.py \
    --in=pipeline_output/pan_printable.stl \
    --out=pipeline_output/pan_printable.stl

step "OPTIONAL  Split into 7 printable pieces (≈ 25 min EXACT boolean × 7)"
echo "  Skipped by default — run manually if you need the parts:"
echo "    \"\$BLENDER\" --background --python split_assembly.py -- pipeline_output/pan_printable.stl"

step "DONE"
ls -la pipeline_output/pan_printable.stl
echo
echo "Inspect:  pipeline_output/pan_printable.stl"
