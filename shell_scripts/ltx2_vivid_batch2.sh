#!/bin/bash
# LTX-2 Batch 2 — Vivid Cinematic II — 1920x1088 @ 161 frames (~6.7s)
# New scenes, first & last frame extraction
# NOTE: no set -e — we want the batch to continue even if one render fails

TOTAL=12
CURRENT=0
START_TIME=$(date +%s)
OUTPUT_DIR=~/ltx2/outputs
BATCH_DIR="$OUTPUT_DIR/batch2"
LOG="$BATCH_DIR/batch2_log.txt"

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
echo "  LTX-2 Batch 2 — $(date)" | tee -a "$LOG"
echo "  Resolution: 1920x1088 | Frames: 161" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# 1. VENETIAN FOG — Pre-dawn canal, lantern reflections
progress "Venetian Fog"
run_render "01_venetian_fog" \
  --prompt "A slow tracking shot glides along a narrow Venetian canal in the final darkness before dawn, dense fog erasing everything beyond twenty feet. A single wrought-iron lantern mounted on a crumbling brick wall casts a trembling sphere of warm amber light across the black water, its reflection stretched into a long vertical brushstroke by tiny ripples. A moored gondola bobs gently at its post, dark lacquered wood gleaming wet. As the camera drifts forward, a second lantern emerges from the fog ahead, creating a receding chain of warm lights swallowed by grey. Condensation beads on every stone surface, catching the glow as pinpoints of gold. 50mm f/2, shallow depth of field, volumetric fog, warm amber against cold blue-grey, Kodak Vision3 250D, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2001

# 2. JUNGLE CANOPY — Overhead descent through rainforest layers
progress "Jungle Canopy"
run_render "02_jungle_canopy" \
  --prompt "A slow jib-down descends through the layered canopy of a dense tropical rainforest. The shot begins in blinding white sky, then pushes through the first ceiling of broad waxy leaves where sunlight fractures into hundreds of individual beams stabbing downward through gaps in the foliage. The camera continues its descent past hanging lianas thick as rope, through clouds of tiny insects illuminated like sparks in the scattered light, until the forest floor appears below, dark and soft with decomposing leaves and enormous bracket fungi glowing faintly amber at their edges. 16mm wide angle, deep focus, dappled volumetric light, hyper-saturated tropical greens against deep shadow, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down \
  --lora-weight-name ltx-2-19b-lora-camera-control-jib-down.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2002

# 3. MERCURY POUR — Macro liquid metal on obsidian
progress "Mercury Pour"
run_render "03_mercury_pour" \
  --prompt "An extreme macro close-up of liquid mercury being poured onto a slab of polished obsidian in a pitch-black studio. The silver metal pools and spreads with impossible surface tension, its mirror-perfect surface reflecting a single circular softbox overhead as a crisp white disc. As the mercury expands it splits into satellite droplets that race outward, each one a tiny convex mirror reflecting a distorted copy of the room. The camera pushes in slowly, the depth of field so shallow that only a razor-thin plane is sharp, mercury beads transitioning from crystalline focus to creamy bokeh spheres within millimeters. Macro lens at f/2.8, extreme shallow depth of field, silver and obsidian black color palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2003

# 4. STORM LIGHTHOUSE — Coastal fury, beacon cutting through rain
progress "Storm Lighthouse"
run_render "04_storm_lighthouse" \
  --prompt "A wide establishing shot of a solitary stone lighthouse perched on jagged black rocks during a violent Atlantic storm at dusk. Massive waves crash against the base, sending white spray thirty feet into the air where wind shears it into horizontal mist. The lighthouse beam sweeps slowly through the scene, a solid cone of gold cutting through sheets of grey rain, illuminating each raindrop as a momentary streak before vanishing into darkness. The sky churns with purple-black clouds lit from below by the last ember of sunset on the horizon. The camera holds perfectly still on a distant tripod, the violence of the storm contrasting with the absolute steadiness of the frame. 35mm f/4, deep focus, dramatic chiaroscuro, desaturated palette with warm gold beacon against cold steel grey, Kodak Ektachrome, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2004

# 5. CELLO PLAYER — Intimate performance, single spotlight
progress "Cello Player"
run_render "05_cello_player" \
  --prompt "A slow dolly-in on a woman playing cello alone on the bare stage of an empty concert hall, a single warm spotlight creating a perfect circle of light around her while the rest of the vast auditorium disappears into absolute black. Her bow arm moves in long fluid strokes, the horsehair catching the light as a thin bright line against the dark body of the instrument. Her eyes are closed, head tilted slightly, shadows pooling in the hollows of her collarbone. Dust motes drift through the spotlight beam in lazy spirals. The polished wood of the cello reveals deep grain patterns and amber varnish reflecting the overhead light. 85mm f/1.4, extreme shallow depth of field, single-source chiaroscuro, warm amber spotlight against absolute black, Kodak Portra 800, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2005

# 6. FROZEN LAKE — Arctic minimalism, ice and sky merge
progress "Frozen Lake"
run_render "06_frozen_lake" \
  --prompt "A wide shot of a perfectly frozen lake stretching to every horizon under a flat overcast Arctic sky, the ice so clear and the clouds so uniform that the boundary between ground and sky dissolves into a seamless field of pale blue-white. A single figure in a dark coat walks slowly away from the camera toward the vanishing point, their footsteps leaving dark marks on the frost-covered surface that form a receding line of punctuation across the emptiness. Fracture lines in the ice radiate outward in geometric patterns beneath the translucent surface, revealing dark water below. The camera tracks forward slowly, following at a distance that never closes. 50mm f/4, deep focus, extreme minimalist composition, desaturated palette of ice blue and slate grey, Fujifilm Provia transparency look, 180-degree shutter, natural motion blur, stabilized tracking." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2006

# 7. SOUK SPICES — Moroccan market corridor, color explosion
progress "Souk Spices"
run_render "07_souk_spices" \
  --prompt "A slow tracking shot moves through a narrow covered spice souk in Marrakech, mountains of ground spices piled in perfect conical pyramids on either side of the passage in vivid saffron yellow, deep paprika red, bright turmeric orange, and dusty green cumin. Shafts of sunlight pierce through gaps in the woven reed ceiling, striking individual spice mounds and igniting their color to fluorescent intensity while neighboring piles fall into cool shadow. Fine spice dust hangs in the light beams like colored smoke. Brass scales and ornate copper vessels line the wooden shelves above. The camera glides forward at waist height through the corridor. 35mm f/2.8, shallow depth of field, hyper-saturated warm palette, Kodak Ektar 100 vivid color, volumetric dust shafts, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2007

# 8. NEON SUBWAY — Empty station, wet tiles, electric color
progress "Neon Subway"
run_render "08_neon_subway" \
  --prompt "A slow dolly-right along an empty subway platform at 3 AM, wet tile walls reflecting strips of magenta and cyan neon in long smeared vertical bands. A thin film of water covers the platform floor, turning it into a mirror that doubles every light source into a shimmering inverted copy. The curved tunnel entrance at the far end of the platform glows deep orange from some distant unseen source, the warm light bleeding along the curved ceiling tiles. A single metal bench sits empty in the middle of the platform, its chrome frame catching every color as prismatic highlights. The camera drifts slowly right, the parallax shifting reflections across every wet surface. 35mm f/2, shallow depth of field, neon magenta and cyan against deep shadow, Kodak Vision3 500T tungsten, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2008

# 9. VOLCANIC GLASS — Obsidian fields, otherworldly landscape
progress "Volcanic Glass"
run_render "09_volcanic_glass" \
  --prompt "A wide tracking shot moves across a vast field of volcanic obsidian in Iceland at blue hour, the glassy black rock surface reflecting the deep cobalt sky like a shattered mirror stretching to the horizon. Thin veins of geothermal steam rise from cracks in the rock, each plume catching the last indigo light and glowing faintly against the darkening sky. In the distance, a snow-capped volcanic peak stands sharp and white against bands of orange and purple along the horizon. The obsidian surface is fractured into angular plates, each tilted at a different angle creating a mosaic of sky reflections. The camera tracks slowly left at knee height, the parallax shifting reflections across thousands of glassy facets. 35mm f/4, deep focus, desaturated blues and volcanic blacks, Fujifilm Velvia transparency saturation, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-left.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2009

# 10. CLOCK WORKSHOP — Macro mechanical movement, brass and glass
progress "Clock Workshop"
run_render "10_clock_workshop" \
  --prompt "An extreme close-up inside an antique clockwork mechanism, the camera positioned between massive brass gears as they turn with glacial precision. Each gear tooth is worn smooth from a century of contact, the metal surface showing a patina of deep gold and verdigris green. A tiny ruby bearing glints blood-red as a steel pivot rotates through it. The camera pushes slowly deeper into the mechanism, racks of progressively smaller gears and springs filling the frame in layers of golden mechanical complexity. A single beam of light from a watchmaker's loupe enters from above, casting spectral rainbows across the brass surfaces. Macro lens at f/4, shallow depth of field, warm brass and gold tones against deep mechanical shadow, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2010

# 11. AURORA CABIN — Northern lights over snow, warm window glow
progress "Aurora Cabin"
run_render "11_aurora_cabin" \
  --prompt "A wide shot of a small wooden cabin in a snow-covered Norwegian valley at midnight, its two windows glowing with warm amber firelight against the deep blue darkness. Above the cabin, the northern lights unfurl in slow undulating curtains of green and violet, their reflection shimmering faintly on the snow-covered ground in pale shifting colors. Dark silhouettes of pine trees frame the scene on both sides, their branches heavy with fresh snow. Stars pierce through gaps in the aurora, sharp and white. Chimney smoke rises in a thin column that catches the green auroral light before dispersing. The camera holds perfectly still on a tripod, the only movement the slow dance of light overhead. 35mm f/2.8, deep focus, rich saturated greens and violets against snow blue, Kodak Portra 800, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2011

# 12. GRAND PIANO — Rain through broken ceiling onto keys
progress "Grand Piano"
run_render "12_grand_piano" \
  --prompt "A slow jib-down descends toward a black grand piano sitting in the center of an abandoned ballroom, its lid propped open to receive a column of rain falling through a gaping hole in the ornate ceiling three stories above. Water streams down in silver threads onto the exposed piano strings, each droplet striking a wire and sending a tiny splash upward that catches the grey daylight filtering through the breach. The piano's polished black surface is covered in a thin sheet of water that reflects the ruined frescoed ceiling above like a dark mirror. Peeling wallpaper and crumbling plaster moldings frame the scene in textures of decay. The camera descends smoothly from ceiling height toward the open piano. 50mm f/2, shallow depth of field, desaturated palette of grey and silver with deep piano black, volumetric rain, Kodak 2383 print stock, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down \
  --lora-weight-name ltx-2-19b-lora-camera-control-jib-down.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 2012

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo ""
echo "==========================================" | tee -a "$LOG"
echo "  Batch 2 complete — 12 renders" | tee -a "$LOG"
echo "  Total time: ${TOTAL_MIN} minutes" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "  Each render: .mp4 + _first.png + _last.png" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
