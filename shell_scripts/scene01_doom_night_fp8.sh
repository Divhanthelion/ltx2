#!/bin/bash
# SCENE 01: "DOOM NIGHT" — Pilot Opening
# 6 clips × ~4 seconds (97 frames @ 24fps = 4.04s each)
# Style: Stylized 3D animation, Pixar/Illumination warmth
# Character: Jake, 12-13, messy hair, oversized headphones, early 2000s bedroom
# NOTE: no set -e — we want the batch to continue even if one render fails

TOTAL=6
CURRENT=0
START_TIME=$(date +%s)
OUTPUT_DIR=~/ltx2/outputs
BATCH_DIR="$OUTPUT_DIR/scene01_doom_night"
LOG="$BATCH_DIR/scene01_log.txt"

mkdir -p "$BATCH_DIR"

if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

sudo -v
SUDO_PID=0
( while true; do sudo -n true; sleep 250; done ) &
SUDO_PID=$!

cleanup() {
    [ "$SUDO_PID" -ne 0 ] && kill "$SUDO_PID" 2>/dev/null
    echo ""
    echo "Scene 01 ended at $(date)" | tee -a "$LOG"
}
trap cleanup EXIT

progress() {
    CURRENT=$((CURRENT + 1))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$CURRENT" -gt 1 ]; then
        AVG=$(( ELAPSED / (CURRENT - 1) ))
        REMAINING=$(( AVG * (TOTAL - CURRENT + 1) ))
        ETA_MIN=$(( REMAINING / 60 ))
        echo "[$CURRENT/$TOTAL] Starting: $1 | ~${ETA_MIN}min remaining" | tee -a "$LOG"
    else
        echo "[$CURRENT/$TOTAL] Starting: $1" | tee -a "$LOG"
    fi
}

extract_and_move() {
    local name="$1"
    local vid
    vid=$(ls -t ${OUTPUT_DIR}/${name}_2*.mp4 2>/dev/null | head -1)
    if [ -z "$vid" ] || [ ! -f "$vid" ]; then
        echo "  No video found matching ${name}_*.mp4 — skipping" | tee -a "$LOG"
        return
    fi
    echo "  Found: $(basename "$vid")" | tee -a "$LOG"
    ffmpeg -y -loglevel error -i "$vid" -vf "select=eq(n\\,0)" -vframes 1 "$BATCH_DIR/${name}_first.png"
    ffmpeg -y -loglevel error -sseof -0.1 -i "$vid" -vframes 1 "$BATCH_DIR/${name}_last.png"
    mv "$vid" "$BATCH_DIR/${name}.mp4"
    rm -f ${OUTPUT_DIR}/${name}_*_latents.npz 2>/dev/null
    rm -f ${OUTPUT_DIR}/${name}_*_frames.npz 2>/dev/null
    echo "  Done: ${name}.mp4 + _first.png + _last.png" | tee -a "$LOG"
}

run_render() {
    local name="$1"
    shift
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1
    sudo docker run --rm --runtime=nvidia --privileged \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v "$OUTPUT_DIR":/outputs ltx2 \
        --output "/outputs/${name}.mp4" \
        "$@"
    extract_and_move "$name"
}

echo "==========================================" | tee "$LOG"
echo "  SCENE 01: DOOM NIGHT — $(date)" | tee -a "$LOG"
echo "  Pilot Opening — 6 clips × 97 frames" | tee -a "$LOG"
echo "  Resolution: 1920x1088 | Style: Animated" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# CLIP 1 — "The Bedroom"
# Slow dolly-in through dark hallway toward cracked bedroom door
# Plants the world: late night, warm cluttered 2000s home, monitor glow
# ---------------------------------------------------------------------------
progress "Clip 1 — The Bedroom"
run_render "clip01_the_bedroom" \
  --prompt "A slow dolly-in through a dark residential hallway at night toward a cracked bedroom door at the far end. Cool blue-white light flickers through the narrow gap in the door, casting shifting rectangles of monitor glow across the opposite wall and hardwood floor. A faded skateboard brand poster hangs on the closed door, barely visible in the darkness. A small warm amber nightlight plugged into a wall outlet near the baseboard provides the only other light source, creating a gentle orange pool at floor level. The hallway walls show framed family photos dissolving into shadow. The camera glides forward at waist height, smooth and steady, drawn toward the flickering light. Stylized 3D animation with soft rounded geometry, warm diffused lighting, and slightly exaggerated proportions. 35mm f/2.8, shallow depth of field, deep navy shadows against warm amber and cool blue highlights, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 97 --seed 6001

# ---------------------------------------------------------------------------
# CLIP 2 — "The Player"
# Close-up on Jake's face lit by monitor glow — we meet our protagonist
# Big expressive eyes, concentration, the glow wrapping around his features
# ---------------------------------------------------------------------------
progress "Clip 2 — The Player"
run_render "clip02_the_player" \
  --prompt "A close-up of a twelve-year-old boy framed from the chest up, lit entirely by the cool blue-white glow of an unseen computer monitor directly in front of him. His messy brown hair falls across his forehead. His wide eyes dart left and right, reflecting tiny points of shifting screen light. His mouth hangs slightly open in deep concentration as he leans forward, gripping something below frame. Oversized silver headphones rest around his neck. Behind him the dark bedroom dissolves into warm shadow — the edge of an unmade bed with rumpled sheets, clothes heaped on the floor, a half-empty glass of milk catching the monitor light on the desk edge. The camera holds steady at the boy's eye level, tripod-locked. Stylized 3D animation with soft diffused lighting, rounded features, big expressive eyes, slightly exaggerated proportions. 50mm f/1.8, shallow depth of field, cool blue key light wrapping around warm skin tones, 180-degree shutter." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 97 --seed 6002

# ---------------------------------------------------------------------------
# CLIP 3 — "The Machine"
# Low angle portrait of the beige PC tower — our hidden second protagonist
# Hold on this: let the audience FEEL the machine's quiet presence
# ---------------------------------------------------------------------------
progress "Clip 3 — The Machine"
run_render "clip03_the_machine" \
  --prompt "A low-angle shot looking upward at a chunky beige desktop PC tower sitting on dark blue carpet beside a wooden desk leg. The tower is large and boxy with rounded plastic edges, early 2000s design. A bright green power LED glows steadily on the front panel. Two smaller amber indicator lights blink in an alternating rhythm. Round colorful stickers decorate the beige side panel — a yellow smiley face, a peeling price tag, a scratched barcode. Faint warm air ripples from the fan vent grille on the lower side. The desk above frames the top of the shot, with dangling headphone cables and a power strip. The camera holds perfectly still, locked low to the ground, framing the PC tower like a character portrait. Stylized 3D animation with warm golden tones on the beige plastic, slightly oversized chunky proportions, soft diffused ambient light. 35mm f/2.8, deep focus, warm amber and gold against cool blue ambient spill from the monitor above, 180-degree shutter, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 97 --seed 6003

# ---------------------------------------------------------------------------
# CLIP 4 — "The Mistake"
# Over-the-shoulder: Jake's fingers hammering keyboard, screen flashing
# Frantic energy — the game is going wrong
# NOTE: LTX-2 can't render actual game screens, so we describe light/color
# ---------------------------------------------------------------------------
progress "Clip 4 — The Mistake"
run_render "clip04_the_mistake" \
  --prompt "An over-the-shoulder shot from behind a boy sitting at a desk, his dark silhouette framed against the bright glow of a computer monitor. His small hands hammer frantically at a chunky beige keyboard in the foreground, fingers blurred with rapid motion. The monitor casts rapidly shifting light across his messy hair and the back of his worn gray t-shirt — flashing red and orange, then white, then red again, the colors of an intense action sequence. His shoulders hunch and tense, his whole body rocking slightly with each keypress. The desk around the keyboard is cluttered with a glass of milk, crumpled papers, and a mouse pad with a faded logo. The camera holds steady behind him at shoulder height, framing both the boy and the flashing screen. Stylized 3D animation with soft rounded proportions, warm room tones, rich saturated screen-light color spill. 50mm f/2, shallow depth of field with the keyboard softly blurred in foreground, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 97 --seed 6004

# ---------------------------------------------------------------------------
# CLIP 5 — "The Death"
# Tight on monitor: screen goes red, fades to dark, Jake's reflection appears
# The moment of defeat — sudden quiet after chaos
# ---------------------------------------------------------------------------
progress "Clip 5 — The Death"
run_render "clip05_the_death" \
  --prompt "A tight close-up filling the entire frame with the curved glass surface of a CRT computer monitor. The screen blazes bright red, then slowly dims and fades through dark crimson to a deep blue-black static glow over four seconds. As the bright colors drain away, the faint ghostly reflection of a young boy becomes visible in the darkening glass — his small rounded silhouette slumping backward in a chair, shoulders dropping in defeat. The monitor's glass surface catches subtle room reflections — the edge of a desk lamp, a distant doorway. The screen settles into a dim steady glow with soft color bands drifting across the surface. The camera holds perfectly still, locked tight on the monitor face, intimate and unflinching. Stylized 3D animation with rich saturated reds fading to cool deep blues, soft glass reflections, slightly curved CRT screen distortion. Macro lens at f/2.8, shallow depth of field, 180-degree shutter." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 97 --seed 6005

# ---------------------------------------------------------------------------
# CLIP 6 — "The Question"
# Wide shot of the whole room. Jake slumped, reaching for mouse.
# THE SEED: Gateway's LED flickers in the last frame. Blink and miss it.
# NOTE: The LED flicker is subtle — LTX-2 may or may not catch it, but the
#        prompt plants the intention. May need compositing in post.
# ---------------------------------------------------------------------------
progress "Clip 6 — The Question"
run_render "clip06_the_question" \
  --prompt "A wide shot from the corner of a dark bedroom showing the entire room. A twelve-year-old boy sits slumped deep in a worn office chair at a wooden desk, his body language heavy and defeated, bathed in the soft blue-white glow of a computer monitor showing a dim static screen. The room around him is warmly cluttered — unmade bed with rumpled blankets against the far wall, clothes scattered on the carpet, posters on the walls fading into darkness. On the floor beside the desk, the chunky beige PC tower sits with its green power LED glowing steadily. The boy slowly reaches his right hand toward the mouse on the desk, lazy and reluctant. The camera holds still from the room corner, framing the full scene like a portrait of solitude and warm exhaustion. Stylized 3D animation with soft rounded geometry, warm navy shadows, cool blue monitor light mixing with deep amber undertones, cozy nostalgic atmosphere. 16mm wide angle, deep focus, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 97 --seed 6006

# ---------------------------------------------------------------------------

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo "" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "  Scene 01 complete — 6 clips" | tee -a "$LOG"
echo "  Total time: ${TOTAL_MIN} minutes" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "POST-PRODUCTION NOTES:" | tee -a "$LOG"
echo "  - Clip 6: Gateway's LED flicker may need compositing" | tee -a "$LOG"
echo "    (single-frame green flash on the PC tower in last 2-3 frames)" | tee -a "$LOG"
echo "  - Audio: All clips need sound design pass" | tee -a "$LOG"
echo "    (game SFX, ambient hum, keyboard clatter, Jake's voice)" | tee -a "$LOG"
echo "  - Continuity check: glass of milk, headphones, PC stickers" | tee -a "$LOG"
