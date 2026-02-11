#!/bin/bash
# LTX-2 LoRA Test — All 7 camera control LoRAs, same prompt
# 1920x1088 @ 161 frames for production-quality comparison
set -e

TOTAL=7
CURRENT=0
START_TIME=$(date +%s)
OUTPUT_DIR="$(pwd)/outputs"
BATCH_DIR="$OUTPUT_DIR/lora_test"
LOG="$BATCH_DIR/lora_test_log.txt"

mkdir -p "$BATCH_DIR"

sudo -v
SUDO_PID=0
( while true; do sudo -n true; sleep 250; done ) &
SUDO_PID=$!

cleanup() {
    [ "$SUDO_PID" -ne 0 ] && kill "$SUDO_PID" 2>/dev/null
    echo ""
    echo "Test batch ended at $(date)" | tee -a "$LOG"
}
trap cleanup EXIT

PROMPT="A wide shot of a weathered stone lighthouse on a rocky cliff at golden hour. Waves crash against the rocks below sending white spray upward. Warm amber light from the setting sun illuminates the lighthouse tower while long shadows stretch across the cliff face. Sea birds drift lazily on thermal currents in the background. The grass on the clifftop sways gently in the ocean breeze. 50mm lens, shallow depth of field, rich saturated color, Kodak 2383 print look, 180-degree shutter, natural motion blur."

run_test() {
    local name="$1"
    local lora_repo="$2"
    local lora_weight="$3"
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] $name" | tee -a "$LOG"
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1
    sudo docker run --rm --runtime=nvidia --privileged \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v "$OUTPUT_DIR":/outputs ltx2 \
        --prompt "$PROMPT" \
        --lora "$lora_repo" --lora-weight-name "$lora_weight" \
        --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 42 \
        --output "/outputs/${name}.mp4"
    local vid
    vid=$(ls -t ${OUTPUT_DIR}/${name}_2*.mp4 2>/dev/null | head -1)
    if [ -n "$vid" ] && [ -f "$vid" ]; then
        mv "$vid" "$BATCH_DIR/${name}.mp4"
        echo "  Done: $name.mp4" | tee -a "$LOG"
    else
        echo "  No video produced for $name" | tee -a "$LOG"
    fi
    rm -f ${OUTPUT_DIR}/${name}_*_latents.npz ${OUTPUT_DIR}/${name}_*_frames.npz 2>/dev/null
}

echo "==========================================" | tee "$LOG"
echo "  LTX-2 LoRA Comparison Test — $(date)" | tee -a "$LOG"
echo "  Resolution: 1920x1088 | Frames: 161 | Seed: 42" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

run_test "dolly_in" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In" "ltx-2-19b-lora-camera-control-dolly-in.safetensors"
run_test "dolly_out" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out" "ltx-2-19b-lora-camera-control-dolly-out.safetensors"
run_test "dolly_left" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left" "ltx-2-19b-lora-camera-control-dolly-left.safetensors"
run_test "dolly_right" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right" "ltx-2-19b-lora-camera-control-dolly-right.safetensors"
run_test "jib_up" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up" "ltx-2-19b-lora-camera-control-jib-up.safetensors"
run_test "jib_down" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down" "ltx-2-19b-lora-camera-control-jib-down.safetensors"
run_test "static" "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static" "ltx-2-19b-lora-camera-control-static.safetensors"

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo "" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "  LoRA test complete — 7 renders" | tee -a "$LOG"
echo "  Total time: ${TOTAL_MIN} minutes" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
