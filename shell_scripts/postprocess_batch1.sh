#!/bin/bash
# Post-process all batch1 videos: interpolate 2x then upscale 2x
# Input:  outputs/batch1/*.mp4
# Output: outputs/batch1_postprocessed/*.mp4
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="$REPO_ROOT/outputs/batch1"
OUTPUT_DIR="$REPO_ROOT/outputs/batch1_postprocessed"

mkdir -p "$OUTPUT_DIR"

VIDEOS=(
    01_noir_rain
    02_deep_sea
    03_golden_hour
    04_abandoned_industrial
    05_macro_alchemy
    06_snow_cabin
    07_desert_highway
    08_underwater_cathedral
    09_tokyo_train
    10_forge
    11_bioluminescent_tide
    12_library_dust
)

TOTAL=${#VIDEOS[@]}
FAILED=()

echo "============================================================"
echo "  Batch 1 Post-Processing: interpolate 2x + upscale 2x"
echo "  $TOTAL videos to process"
echo "============================================================"
echo ""

for i in "${!VIDEOS[@]}"; do
    NAME="${VIDEOS[$i]}"
    NUM=$((i + 1))
    INPUT="$INPUT_DIR/${NAME}.mp4"
    OUTPUT="$OUTPUT_DIR/${NAME}.mp4"

    echo ""
    echo "************************************************************"
    echo "  [$NUM/$TOTAL] $NAME"
    echo "************************************************************"

    if [ ! -f "$INPUT" ]; then
        echo "  SKIP: $INPUT not found"
        FAILED+=("$NAME (not found)")
        continue
    fi

    if [ -f "$OUTPUT" ]; then
        echo "  SKIP: $OUTPUT already exists"
        continue
    fi

    START_TIME=$SECONDS

    "$SCRIPT_DIR/postprocess.sh" "$INPUT" "$OUTPUT" --interpolate 2x --upscale 2x

    ELAPSED=$(( SECONDS - START_TIME ))
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))
    echo "  Completed in ${MINS}m ${SECS}s"

    if [ ! -f "$OUTPUT" ]; then
        FAILED+=("$NAME (output missing)")
    fi
done

echo ""
echo "============================================================"
echo "  BATCH COMPLETE"
echo "============================================================"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All $TOTAL videos processed successfully!"
else
    echo "  Failed: ${#FAILED[@]}/$TOTAL"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi

echo ""
echo "  Output: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"*.mp4 2>/dev/null || echo "  (no output files)"
