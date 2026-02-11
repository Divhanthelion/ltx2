#!/bin/bash
# LTX-2 Batch 4 — PUPPY CHAOS — 1920x1088 @ 161 frames (~6.7s)
# Cute chaos, puppies being absolute gremlins
# NOTE: no set -e — we want the batch to continue even if one render fails

TOTAL=12
CURRENT=0
START_TIME=$(date +%s)
OUTPUT_DIR="$(pwd)/outputs"
BATCH_DIR="$OUTPUT_DIR/batch4_puppies"
LOG="$BATCH_DIR/batch4_log.txt"

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
echo "  LTX-2 Batch 4 — PUPPY CHAOS — $(date)" | tee -a "$LOG"
echo "  Resolution: 1920x1088 | Frames: 161" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# 1. TOILET PAPER TORNADO — Golden retriever puppy destroys bathroom
progress "Toilet Paper Tornado"
run_render "01_toilet_paper_tornado" \
  --prompt "A low-angle shot from bathroom floor level of a golden retriever puppy sprinting in tight circles around a small bathroom, a long trail of unraveled toilet paper streaming behind it like a victory banner. The puppy's oversized paws skid on the tile floor as it corners, ears flopping wildly, tongue hanging sideways from its open mouth in pure joy. Shredded white toilet paper drapes over the bathtub edge, the toilet seat, and the sink vanity in chaotic loops. The camera holds still on the floor, the puppy racing past in a blur of blonde fur and white streamers. Bright bathroom overhead light, warm golden fur against white tile, 35mm f/2.8, shallow depth of field, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4001

# 2. MUD MONSTER — Husky puppy post-puddle, zero remorse
progress "Mud Monster"
run_render "02_mud_monster" \
  --prompt "A medium close-up of a husky puppy sitting proudly on a white kitchen floor, absolutely caked in dark brown mud from nose to tail. Its bright blue eyes peer out from a mud-covered face with an expression of complete satisfaction. Muddy paw prints trail across the pristine white tile behind it in a chaotic path from the open back door, where a destroyed flower bed is faintly visible in the sunny garden beyond. The puppy's thick fur is matted and dripping, a small puddle of muddy water forming beneath it. The camera pushes in slowly. Bright kitchen daylight, extreme contrast between the immaculate white kitchen and the muddy puppy, 50mm f/2, shallow depth of field, Kodak Portra warmth, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4002

# 3. COUCH CRIME SCENE — Corgi puppy inside a destroyed pillow
progress "Couch Crime Scene"
run_render "03_couch_crime_scene" \
  --prompt "A wide shot of a living room where a small corgi puppy sits in the exact center of a devastated couch, surrounded by an explosion of white pillow stuffing that covers every surface like fake snow. The puppy has a large tuft of stuffing stuck to the top of its head and another dangling from its mouth, chewing slowly with a blissful distant expression. Ripped pillow fabric lies in shreds across the cushions. Warm afternoon light pours through the living room window, catching the floating stuffing particles like gentle snowfall. The camera holds steady, framing the full scope of the destruction. 35mm f/2.8, warm afternoon light, cozy living room palette covered in white chaos, Kodak Portra, 180-degree shutter, natural motion blur, tripod-locked." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4003

# 4. SPRINKLER ATTACK — Beagle puppy leaping into garden sprinkler
progress "Sprinkler Attack"
run_render "04_sprinkler_attack" \
  --prompt "A low tracking shot follows a beagle puppy charging across a sunlit backyard lawn toward an oscillating garden sprinkler, ears streaming behind it like flags. The puppy leaps mouth-first into the arc of water, snapping at individual streams with its jaws, water spraying in every direction and catching the afternoon sunlight as thousands of tiny golden droplets. Its compact body twists mid-stride, paws splashing through the waterlogged grass, sending small fans of water up from each footfall. The camera tracks alongside at puppy height, keeping pace with the charge. Bright summer afternoon, golden backlit water droplets against deep green grass, 35mm f/2, shallow depth of field, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4004

# 5. SHOE THIEF — Dachshund puppy dragging a boot bigger than itself
progress "Shoe Thief"
run_render "05_shoe_thief" \
  --prompt "A slow dolly-in on a miniature dachshund puppy determinedly dragging a large leather hiking boot across a hardwood hallway floor, the boot easily twice the puppy's size. The puppy grips the boot's tongue in its tiny jaws and walks backward with stiff determined legs, its long body stretched out, nails clicking on the wood floor as it hauls its prize inch by inch. The puppy's tail wags furiously despite the enormous physical effort, ears bouncing with each backward step. Warm afternoon light streams through a side window, casting long golden shadows across the hallway floor. The camera pushes in slowly from the far end of the hall. 50mm f/2, shallow depth of field, warm golden hardwood tones, Kodak Portra palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4005

# 6. LEAF PILE AMBUSH — Lab puppy erupting from autumn leaves
progress "Leaf Pile Ambush"
run_render "06_leaf_pile_ambush" \
  --prompt "A wide shot of a massive pile of raked autumn leaves in a suburban backyard under golden late afternoon light. The pile erupts as a chocolate labrador puppy bursts upward from inside, leaves scattering in every direction, the puppy launching itself into the air with all four legs extended, mouth open in a wide panting grin, orange and red leaves cascading around it in slow arcs. Golden hour sunlight backlights the airborne leaves as translucent stained glass in amber and crimson. The puppy hangs at the apex of its leap, suspended in a cloud of autumn color. The camera holds steady on a tripod, capturing the full explosion. 35mm f/2.8, shallow depth of field, saturated autumn palette of red orange and gold, Kodak Ektar vivid color, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4006

# 7. BATH TIME ESCAPE — Wet French bulldog puppy on the run
progress "Bath Time Escape"
run_render "07_bath_time_escape" \
  --prompt "A low tracking shot follows a soaking wet French bulldog puppy tearing down a residential hallway at maximum speed, its compact body leaving a trail of wet paw prints and water splatter on the hardwood floor. Suds of white shampoo cling to its head and back in foamy patches, and its bat-like ears are plastered flat against its skull. The puppy's stubby legs are a blur of motion, its entire body wiggling with each stride as water droplets fly off its slick coat. Behind it, a bathroom doorway spills light and steam, a toppled shampoo bottle visible on the wet tile floor. The camera tracks alongside at floor level, barely keeping up. 35mm f/2, shallow depth of field, warm indoor lighting, water droplets catching the light, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4007

# 8. GARDEN DIGGER — Border collie puppy excavating a flower bed
progress "Garden Digger"
run_render "08_garden_digger" \
  --prompt "A medium shot of a border collie puppy digging frantically in a well-maintained flower bed, dirt flying in a continuous fountain from between its back legs. The puppy's front paws are a blur of motion, black and white fur now brown to the elbows, its nose buried in the growing crater. Uprooted marigolds and petunias lie scattered on the lawn behind it, roots exposed, soil clinging to their stems. The hole is already impressively deep for such a small animal. Bright overcast light, even and soft, rich garden greens against dark turned earth. The camera holds at a slight downward angle, framing the full excavation in progress. 50mm f/2.8, moderate depth of field, vivid garden palette, 180-degree shutter, natural motion blur, tripod-locked." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4008

# 9. ZOOMIES AT DUSK — Pomeranian puppy doing laps in a yard
progress "Zoomies at Dusk"
run_render "09_zoomies_dusk" \
  --prompt "A wide shot of a tiny cream Pomeranian puppy performing full-speed zoomies across a suburban backyard at dusk, a puffball of fur racing in enormous figure-eights across the dewy grass. The puppy's legs move so fast they nearly disappear, its fluffy body hovering above the ground like a small furry hovercraft, tongue trailing sideways from its grinning mouth. Each sharp turn sends a tiny spray of dew droplets arcing through the deep blue twilight air. String lights along the back fence provide warm amber bokeh circles in the background. The camera pans gently to follow the puppy's chaotic laps. 50mm f/1.8, extreme shallow depth of field, warm string light bokeh against blue hour sky, cream-colored fur catching the last ambient light, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4009

# 10. KITCHEN COUNTER HEIST — Puppy stretching for food on counter
progress "Kitchen Counter Heist"
run_render "10_counter_heist" \
  --prompt "A medium shot from across a bright kitchen of a lanky German shepherd puppy standing on its hind legs against the kitchen counter, front paws on the granite edge, stretching its neck to maximum extension toward a plate of sandwiches just barely out of reach. Its nose twitches rapidly, tongue extending to its absolute limit trying to reach the nearest sandwich. The puppy's back legs tremble with the effort, hind paws barely touching the tile floor. Its tail wags in tight rapid circles of anticipation. Bright midday kitchen light floods through the window behind the counter, backlighting the puppy's fuzzy oversized ears as translucent pink-gold halos. The camera holds still. 35mm f/2.8, bright natural kitchen light, warm domestic palette, Kodak Portra, 180-degree shutter, natural motion blur, tripod-locked." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4010

# 11. SNOW ZOOMIES — Samoyed puppy bounding through fresh snow
progress "Snow Zoomies"
run_render "11_snow_zoomies" \
  --prompt "A wide tracking shot of a fluffy white Samoyed puppy bounding through fresh deep snow in a quiet suburban yard, disappearing almost entirely with each leap and then exploding upward in a burst of powder. Its pure white fur is nearly invisible against the snow except for its black nose and dark sparkling eyes. Each bound sends up a plume of fine powder that hangs and glitters in the low winter sunlight. The puppy's entire body arcs through the air between leaps, ears pinned back, mouth open in a wide grin. The camera tracks alongside, following the bouncing white shape through the white landscape. Cold bright morning light, soft blue shadows on snow, 35mm f/2.8, moderate depth of field, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4011

# 12. FOOD COMA — Aftermath of chaos, puppy passed out cold
progress "Food Coma"
run_render "12_food_coma" \
  --prompt "A slow dolly-in on a small bulldog puppy lying completely unconscious on its back on a living room rug, all four stubby legs pointing straight up at the ceiling, pink belly rising and falling with deep slow breaths. Its mouth hangs slightly open, a tiny thread of drool connecting its jowl to the rug. Around it lies the aftermath of total destruction: a chewed-up TV remote, scattered kibble, a demolished squeaky toy leaking white stuffing, and a toppled water bowl creating a dark wet circle on the carpet. Warm late afternoon light slants through the window, casting a golden beam directly across the sleeping puppy. The camera pushes in slowly and gently. 85mm f/1.8, extreme shallow depth of field, warm golden afternoon light, cozy domestic palette, Kodak Portra, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 4012

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo ""
echo "==========================================" | tee -a "$LOG"
echo "  Batch 4 PUPPY CHAOS complete — 12 renders" | tee -a "$LOG"
echo "  Total time: ${TOTAL_MIN} minutes" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
