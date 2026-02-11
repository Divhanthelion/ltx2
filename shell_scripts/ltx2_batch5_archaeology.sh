#!/bin/bash
# LTX-2 Batch 5 — ANCIENT SITES — 1920x1088 @ 161 frames (~6.7s)
# NOTE: no set -e — we want the batch to continue even if one render fails

TOTAL=16
CURRENT=0
START_TIME=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$REPO_ROOT/outputs"
BATCH_DIR="$OUTPUT_DIR/batch5_archaeology"
LOG="$BATCH_DIR/batch5_log.txt"

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
    echo "Batch ended at $(date)" | tee -a "$LOG"
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
echo "  LTX-2 Batch 5 — ANCIENT SITES — $(date)" | tee -a "$LOG"
echo "  Resolution: 1920x1088 | Frames: 161" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# 1. STONEHENGE — Dawn mist rolling between the stones
progress "Stonehenge"
run_render "01_stonehenge" \
  --prompt "A slow dolly-in toward Stonehenge at dawn as pale golden light breaks across the Salisbury Plain. The massive sarsen stones stand in their ancient circle, their weathered grey surfaces catching the first warm rays while long shadows stretch across dewy grass. A low ground mist curls between the trilithons, glowing amber where sunlight passes through it. The camera pushes forward steadily at waist height, the stones growing larger and more imposing as the central altar stone comes into view. 50mm lens, shallow depth of field, warm golden light against cool blue shadows, Kodak Vision3 250D, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5001

# 2. GREAT PYRAMID OF GIZA — Sunset grandeur
progress "Great Pyramid"
run_render "02_great_pyramid" \
  --prompt "A wide establishing shot of the Great Pyramid of Giza at golden hour, its massive limestone blocks glowing warm amber in the low desert sun. The pyramid rises in perfect geometric precision against a deep orange and purple sky, each course of stone casting a thin shadow line across the face below it. Desert sand stretches flat to the horizon in every direction. A warm dry wind carries fine sand particles that catch the light as golden streaks. The camera holds perfectly still on a tripod, the pyramid monumental and timeless in the frame. 35mm f/4, deep focus, warm desert palette of amber and gold against deepening blue sky, Kodak Ektachrome, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5002

# 3. GOBEKLI TEPE — Ancient carved pillars
progress "Gobekli Tepe"
run_render "03_gobekli_tepe" \
  --prompt "A slow dolly-right through the excavated circular enclosure of Gobekli Tepe, the world's oldest temple. Massive T-shaped limestone pillars stand in their original positions, their flat surfaces carved with intricate relief sculptures of foxes, vultures, and scorpions weathered by twelve thousand years. Morning light rakes across the carved surfaces at a low angle, revealing every chisel mark and erosion pattern in sharp relief. The surrounding earth walls show clean archaeological excavation layers in bands of tan and ochre. The camera glides past each pillar, revealing the carvings one by one. 35mm f/2.8, shallow depth of field, warm morning light on ancient limestone, documentary aesthetic, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5003

# 4. COLOSSEUM — Interior grandeur
progress "Colosseum"
run_render "04_colosseum" \
  --prompt "A slow dolly-in through an archway into the interior of the Roman Colosseum, revealing the vast oval arena and tiered seating stretching upward in concentric rings of weathered travertine stone. Warm afternoon sunlight pours through the open side, casting dramatic angular shadows across the exposed underground hypogeum chambers visible below the arena floor. Fragments of marble facing still cling to sections of wall. The scale is overwhelming, each archway framing another vista of ancient engineering. The camera pushes forward through the entrance arch, the arena opening up before it. 50mm lens, shallow depth of field, warm Roman light, Kodak Vision3 500T, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5004

# 5. MACHU PICCHU — Cloud forest reveal
progress "Machu Picchu"
run_render "05_machu_picchu" \
  --prompt "A slow jib-up reveals Machu Picchu emerging from morning clouds clinging to the Andean peaks. The camera rises to reveal precisely fitted stone walls and terraces cascading down the mountain ridge, the iconic Huayna Picchu peak looming behind in dramatic silhouette. Clouds drift through the ruins at ground level, partially obscuring and then revealing stone doorways and ceremonial plazas. Bright green grass covers the terraces between grey granite walls. The vast Urubamba valley drops away thousands of feet below into blue haze. 35mm f/4, deep focus, cool mountain light with warm stone tones, Kodak Portra palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up \
  --lora-weight-name ltx-2-19b-lora-camera-control-jib-up.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5005

# 6. PETRA — The Treasury through the Siq canyon
progress "Petra Treasury"
run_render "06_petra_treasury" \
  --prompt "A slow dolly-in through the narrow Siq canyon toward the Treasury of Petra, the towering sandstone walls on either side carved by water into flowing organic shapes in bands of rose pink, salmon, cream and rust. The canyon narrows to a thin slice of sky above. Ahead, framed perfectly by the canyon walls, the iconic facade of Al-Khazneh emerges in warm golden light, its Hellenistic columns and pediments carved directly into the living rock face. The camera pushes forward through the shadowed canyon, the Treasury facade growing brighter and larger with each step. 50mm lens, shallow depth of field, rose and gold sandstone palette, Kodak Vision3 250D, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5006

# 7. ANGKOR WAT — Sunrise reflection in the moat
progress "Angkor Wat"
run_render "07_angkor_wat" \
  --prompt "A wide static shot of Angkor Wat at sunrise, the five lotus-shaped towers perfectly reflected in the still water of the rectangular moat in the foreground. The sky transitions from deep indigo to warm orange behind the temple silhouette. Palm trees line the far shore as dark shapes against the brightening sky. The reflection shimmers with tiny ripples as the morning breeze begins. The camera holds perfectly still, the temple and its mirror image filling the frame symmetrically. 35mm f/4, deep focus, sunrise color palette from indigo to gold, Kodak Ektachrome, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5007

# 8. CHICHEN ITZA — El Castillo pyramid
progress "Chichen Itza"
run_render "08_chichen_itza" \
  --prompt "A slow dolly-out from the base of El Castillo at Chichen Itza, the great stepped pyramid rising in perfect symmetry above, each of its nine terraces casting a sharp shadow line across the one below. The precisely carved limestone steps ascend steeply to the temple platform at the summit. The camera pulls back slowly from the base, revealing the full monumental scale of the pyramid against a deep blue tropical sky with towering white cumulus clouds. Bright green grass of the Great Plaza stretches flat around the base. 35mm f/4, deep focus, warm tropical light on pale limestone, Kodak Portra palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-out.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5008

# 9. EASTER ISLAND — Moai statues at dusk
progress "Easter Island Moai"
run_render "09_easter_island" \
  --prompt "A slow dolly-right along the row of fifteen restored Moai statues at Ahu Tongariki on Easter Island at dusk. Each massive stone figure stands in solemn profile, their elongated heads and prominent brows silhouetted against a dramatic sky of deep purple and orange. The camera tracks laterally past each statue, their weathered volcanic tuff surfaces catching the last warm light on their faces while their backs fall into cool shadow. Wild grass sways at their base. The Pacific Ocean stretches to the horizon behind them. 50mm lens, shallow depth of field, dramatic dusk silhouette, Kodak Vision3 500T, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5009

# 10. PARTHENON — Golden hour colonnade
progress "Parthenon"
run_render "10_parthenon" \
  --prompt "A slow dolly-right along the colonnade of the Parthenon in Athens at golden hour. Massive fluted Doric columns of Pentelic marble glow warm honey-gold in the low sunlight, each column casting a long shadow across the stylobate. The camera glides past column after column, the rhythmic spacing creating a hypnotic parallax effect as the distant Athenian cityscape shifts behind them. Fragments of the original marble frieze are visible along the entablature above. The Acropolis plateau drops away to reveal Athens sprawling below in hazy golden light. 50mm lens, shallow depth of field, warm marble tones, Kodak Vision3 250D, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5010

# 11. POMPEII — Frozen Roman street
progress "Pompeii"
run_render "11_pompeii" \
  --prompt "A slow dolly-in down an ancient Roman street in Pompeii, the original paving stones worn smooth by centuries of cart wheels, deep ruts still visible in the volcanic rock. Ruined walls rise on either side, fragments of frescoes in faded reds and yellows still clinging to patches of plaster. Stepping stones cross the street at intervals. Mount Vesuvius looms in the distance at the end of the street, its slopes green and deceptively peaceful under a blue sky. Wildflowers grow from cracks in the ancient masonry. The camera pushes forward at eye level down the center of the street. 50mm lens, shallow depth of field, warm Mediterranean light, documentary aesthetic, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5011

# 12. GREAT WALL — Misty mountain ridges
progress "Great Wall"
run_render "12_great_wall" \
  --prompt "A wide static shot of the Great Wall of China snaking across mountain ridges in the early morning mist. The wall traces the contours of steep green mountains, rising and falling with the terrain, its watchtowers appearing and disappearing as bands of white cloud drift across the peaks. The stonework is weathered grey-brown, partially overgrown with moss where the forest encroaches. Morning light filters through the mist creating soft volumetric shafts between the mountain ridges. The camera holds still, the wall stretching into misty infinity in both directions. 35mm f/4, deep focus, misty mountain greens and greys, Kodak Portra palette, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5012

# 13. TIKAL — Jungle temple rising above canopy
progress "Tikal"
run_render "13_tikal" \
  --prompt "A slow jib-up from the jungle canopy reveals the towering limestone pyramid of Temple I at Tikal rising above the treetops of the Guatemalan rainforest. The camera rises through layers of green canopy until the steep pyramid emerges dramatically against a hazy tropical sky. The roofcomb at its summit is elaborately carved. Dense jungle stretches unbroken to the horizon in every direction, other temple pyramids poking above the canopy in the distance. 35mm f/4, deep focus, lush jungle greens against pale limestone, Kodak Vision3 250D, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up \
  --lora-weight-name ltx-2-19b-lora-camera-control-jib-up.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5013

# 14. ABU SIMBEL — Colossal facade of Ramesses II
progress "Abu Simbel"
run_render "14_abu_simbel" \
  --prompt "A slow dolly-in toward the colossal facade of Abu Simbel, four seated figures of Ramesses II carved directly into the sandstone cliff, each over sixty feet tall. The morning sun illuminates the left pair while the right pair remains in cool shadow, the play of light revealing extraordinary detail in the carved headdresses, false beards and hieroglyphic cartouches. Smaller figures of family members stand between the massive legs. The dark rectangular entrance to the temple opens at the center. The camera pushes forward steadily, the colossal figures growing ever larger in the frame. 50mm lens, shallow depth of field, warm Egyptian light on sandstone, Kodak Ektachrome, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5014

# 15. NEWGRANGE — Passage tomb entrance
progress "Newgrange"
run_render "15_newgrange" \
  --prompt "A slow dolly-in toward the entrance of Newgrange passage tomb in Ireland, the massive white quartz facade gleaming in overcast light. The entrance stone lies horizontally before the dark doorway, its surface covered in the famous triple spiral carvings cut deep into grey granite five thousand years ago. The grassy mound rises behind, a great dome of earth ringed by kerbstones. The narrow stone-lined passage disappears into darkness beyond the entrance. Soft Irish rain mists the air. Green rolling hills stretch into grey distance beyond. The camera pushes forward slowly toward the carved entrance stone. 50mm lens, shallow depth of field, overcast silver light on white quartz, Kodak Portra palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5015

# 16. BAALBEK — Temple of Jupiter columns
progress "Baalbek"
run_render "16_baalbek" \
  --prompt "A slow dolly-out from the base of the six remaining columns of the Temple of Jupiter at Baalbek in Lebanon, each column over sixty feet tall and seven feet in diameter, the largest Roman columns ever erected. The camera pulls back and tilts up to reveal their full impossible height, the ornate Corinthian capitals and the massive stone entablature they still support. The trilithon stones in the foundation wall are visible at the base, each weighing over eight hundred tons. Late afternoon light rakes across the fluted column shafts in warm gold. Blue Lebanese mountains rise in the distance. 35mm f/2.8, deep focus, warm golden stone against blue sky, Kodak Vision3 250D, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-out.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 5016

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo "" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "  Batch 5 ANCIENT SITES complete — 16 renders" | tee -a "$LOG"
echo "  Total time: ${TOTAL_MIN} minutes" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
