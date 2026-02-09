#!/bin/bash
# Post-production wrapper for RIFE + Real-ESRGAN
# Runs in Docker — does not touch system Python
#
# Setup (one time):
#   docker build -f Dockerfile.postprocess -t ltx2-postprocess .
#
# Usage:
#   ./postprocess.sh input.mp4 output.mp4 [OPTIONS]
#
# Examples:
#   # Smooth out motion (2x frame interpolation, 24→48fps)
#   ./postprocess.sh outputs/video.mp4 outputs/video_smooth.mp4 --interpolate 2x
#
#   # Smooth + re-encode at 60fps
#   ./postprocess.sh outputs/video.mp4 outputs/video_60fps.mp4 --interpolate 4x --target-fps 60
#
#   # Upscale 1080p → 4K
#   ./postprocess.sh outputs/video.mp4 outputs/video_4k.mp4 --upscale 2x
#
#   # Both: smooth then upscale
#   ./postprocess.sh outputs/video.mp4 outputs/video_final.mp4 --interpolate 2x --upscale 2x

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 INPUT OUTPUT [--interpolate 2x|4x] [--upscale 2x|4x] [--target-fps N]"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4 smooth.mp4 --interpolate 2x"
    echo "  $0 video.mp4 final.mp4 --interpolate 2x --upscale 2x"
    exit 1
fi

INPUT=$(realpath "$1")
OUTPUT_DIR=$(dirname "$(realpath -m "$2")")
OUTPUT_NAME=$(basename "$2")
shift 2

# Check image exists
if ! docker image inspect ltx2-postprocess &>/dev/null; then
    echo "Building postprocess image (one-time setup)..."
    docker build -f "$(dirname "$0")/Dockerfile.postprocess" -t ltx2-postprocess "$(dirname "$0")"
fi

docker run --rm --runtime=nvidia \
    -v "$(dirname "$INPUT")":/input:ro \
    -v "$OUTPUT_DIR":/output \
    ltx2-postprocess \
    --input "/input/$(basename "$INPUT")" \
    --output "/output/$OUTPUT_NAME" \
    "$@"
