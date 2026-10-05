#!/usr/bin/env bash
# =============================================================================
# Cheap diagnostic: is training-time PS-normal supervision worth building for
# Gaussian Wrapping, or does post-hoc baking already cover what you need?
#
# Runs stock (unmodified) Gaussian Wrapping, renders its mesh's own normals
# from every COLMAP viewpoint, independently solves photometric stereo from
# the same rig's multi-light captures, and compares the two with the
# disagreement split into low/mid-frequency (actionable -- the algorithm is
# getting something wrong at a scale the mesh could represent) versus
# high-frequency (expected -- finer than any mesh at this resolution could
# ever show, regardless of algorithm).
# =============================================================================
set -euo pipefail

COLMAP_SPARSE_TXT="${1:?usage: run_diagnostic.sh <colmap_sparse_txt_dir> <colmap_dataset_for_gw> <lit_images_dir> <calibration.json> <out_dir>}"
COLMAP_DATASET="${2:?}"
LIT_IMAGES="${3:?}"
CALIBRATION="${4:?}"
OUT_DIR="${5:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GW_OUT="${OUT_DIR}/gaussian_wrapping"
RENDER_OUT="${OUT_DIR}/rendered"
DIAG_OUT="${OUT_DIR}/diagnostic"
CAMERAS_JSON="${OUT_DIR}/cameras.json"

mkdir -p "$GW_OUT" "$RENDER_OUT" "$DIAG_OUT"

echo "[1/4] stock Gaussian Wrapping reconstruction (no modifications)"
python gaussian_wrapping/scripts/train_and_extract_gw_ours.py \
    -s "$COLMAP_DATASET" -m "$GW_OUT" -r 2

# Confirm this matches your actual output filename -- the repo's exact
# naming wasn't independently verified against a real run; adjust if
# train_and_extract_gw_ours.py wrote something else into $GW_OUT.
MESH="${GW_OUT}/mesh_texture_refined.ply"
[ -f "$MESH" ] || MESH="${GW_OUT}/mesh.ply"
[ -f "$MESH" ] || { echo "no mesh found in ${GW_OUT}, check the actual filename"; exit 1; }
echo "using mesh: ${MESH}"

echo "[2/4] extracting camera poses from the COLMAP sparse model"
python "${SCRIPT_DIR}/build_camera_list.py" \
    --sparse-txt "$COLMAP_SPARSE_TXT" --out "$CAMERAS_JSON"

echo "[3/4] rendering mesh normal/depth per view"
blender --background --factory-startup \
    --python "${SCRIPT_DIR}/render_mesh_normals.py" -- \
    --mesh "$MESH" --cameras "$CAMERAS_JSON" --out "$RENDER_OUT"

echo "[4/4] photometric-stereo solve + frequency-separated comparison"
python "${SCRIPT_DIR}/ps_disagreement_check.py" \
    --lit-images "$LIT_IMAGES" \
    --calibration "$CALIBRATION" \
    --rendered "$RENDER_OUT" \
    --cameras "$CAMERAS_JSON" \
    --out "$DIAG_OUT"

echo "done -- see ${DIAG_OUT}/summary.json and the *_lowfreq.png heatmaps"
