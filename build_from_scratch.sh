#!/usr/bin/env bash
# Build the entire DEFINITIVE deepPan model from scratch, including:
#   - pan_printable.{obj,stl}            : fused, watertight, printable pan body
#   - pan_piece_central.stl              : central piece (1)
#   - pan_piece_outer_0..5.stl           : 6 outer 60° sectors with half-wedges
#   - bottom_plate.stl                   : full plate (for visualisation only)
#   - bottom_plate_sector_0..5.stl       : 6 plate sectors (P1S-printable)
#   - plug.stl                           : alignment-peg (4 × 18 mm)
#   - 3mf/*.3mf                          : 14 Bambu P1S build-plates
#   - assembly_view_with_plugs.{obj,stl} : multi-object assembly view
#
# Inputs:
#   data/Tenor Pan only.obj
#   data/pan_centroid_offset.json
#   data/notepads/notepad_properties.json
#
# Run end-to-end (≈ 90 min total):
#   ./build_from_scratch.sh
#
# Then verify:
#   python3 test_pipeline.py        # quick checksum check against MANIFEST
#   python3 test_pipeline.py --full # full regenerate-and-compare (~90 min)

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

step "10/14  Split body into 7 printable pieces (truncates body at Z=-122, solidifies drum wall, builds strut wedges with joinery, EXACT booleans — ~25 min)"
"$BLENDER" --background --python split_assembly.py -- --solver=EXACT

step "11/14  Generate plug.stl (4 × 18 mm horizontal alignment peg)"
python3 generate_plug.py

step "12/14  Generate bottom plate + split into 6 P1S-printable sectors"
python3 generate_bottom_plate.py
python3 split_bottom_plate.py

step "13/14  Generate Bambu P1S .3mf build plates"
python3 generate_3mf.py

step "14/14  Generate multi-object assembly view (no outer sleeves, per-pad mounts)"
python3 generate_assembly_view_with_plugs.py

step "DONE"
ls -la pipeline_output/pan_printable.stl pipeline_output/3mf/
echo
echo "Inspect the assembled model:  pipeline_output/assembly_view_with_plugs.stl"
echo "Open .3mf files in Bambu Studio: pipeline_output/3mf/"
echo
echo "Verify against committed reference:"
echo "  python3 test_pipeline.py            # quick checksum check"
echo "  python3 test_pipeline.py --full     # full regenerate-and-compare"
