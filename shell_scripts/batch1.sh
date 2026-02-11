#!/bin/bash
# LTX-2 Batch 1 — Vivid Cinematic — 1920x1088 @ 161 frames (~6.7s)
# Diverse moods, genres, and camera movements
# NOTE: no set -e — we want the batch to continue even if one render fails

TOTAL=12
CURRENT=0
START_TIME=$(date +%s)
OUTPUT_DIR="$(pwd)/outputs"
BATCH_DIR="$OUTPUT_DIR/batch1"
LOG="$BATCH_DIR/batch1_log.txt"

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
echo "  LTX-2 Batch 1 — $(date)" | tee -a "$LOG"
echo "  Resolution: 1920x1088 | Frames: 161" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# 1. NOIR RAIN — Taipei back alley, neon-soaked noir
progress "Noir Rain"
run_render "01_noir_rain" \
  --prompt "A slow dolly-in through a rain-soaked alleyway in Taipei at 2 AM. Neon signs in Mandarin reflect off wet cobblestones in smeared reds and electric blues. Steam rises from a street vendor's abandoned cart in the midground, catching the colored light like a curtain of silk. A lone figure in a dark raincoat stands motionless under a single flickering fluorescent tube, face obscured by shadow. The camera pushes forward at knee height, stabilized and smooth, tiny water droplets catching light at the lens edge. 35mm f/2.8, shallow depth of field, Kodak Vision3 500T tungsten film stock, deep contrast between warm neon and cold blue shadow, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 101

# 2. DEEP SEA — Abyssal bioluminescence, hydrothermal vent
progress "Deep Sea"
run_render "02_deep_sea" \
  --prompt "A slow tracking shot glides just above the ocean floor at abyssal depth, where absolute darkness gives way to scattered points of bioluminescent life. Translucent deep-sea worms pulse with electric blue veins along a hydrothermal vent crusted in bright orange mineral deposits. Particles of marine snow drift lazily through the frame like inverted snowfall, each fleck catching the faint organic glow. Volumetric caustics ripple across every surface. The camera drifts forward and to the right, revealing the full scale of the vent chimney billowing dark mineral-rich water. 50mm macro lens, extreme shallow depth of field, desaturated blues and blacks punctuated by vivid organic color, 180-degree shutter, natural motion blur, raw documentary footage aesthetic." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 202

# 3. GOLDEN HOUR SILHOUETTE — Rooftop moment, wind and light
progress "Golden Hour Silhouette"
run_render "03_golden_hour" \
  --prompt "A woman in her late twenties stands at the edge of a concrete rooftop in the final ten minutes before sunset, her back to the camera. Horizontal amber light catches every loose strand of her dark hair as wind lifts it across the frame in slow graceful arcs. She holds a half-finished cigarette between two fingers at her side, smoke curling upward in a thin blue-gray thread against the deep orange sky. The city skyline behind her dissolves into a wash of purple and molten gold, windows flashing with reflected sun. The camera holds perfectly still, tripod-locked, framing her from waist up as a dark silhouette against the burning sky. 85mm f/1.4, extreme shallow depth of field, rich saturated Kodak Portra palette, golden hour backlight creating a soft halo, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 303

# 4. ABANDONED INDUSTRIAL — Post-apocalyptic decay, nature reclaiming
progress "Abandoned Industrial"
run_render "04_abandoned_industrial" \
  --prompt "A slow tracking shot moves through the ruins of an enormous abandoned steel mill, where decades of neglect have let nature stage a full invasion. Thick green vines snake up rusted blast furnaces three stories tall, and ferns burst from cracks in the concrete floor. Shafts of late afternoon light pour through holes in the collapsed roof, each beam a solid column of golden dust particles hanging motionless in the still air. Water drips from corroded overhead pipes into shallow pools that reflect the skeletal iron framework above. The camera glides forward at chest height, stabilized, passing between towering columns of oxidized steel stained with streaks of orange and black. 35mm f/2.8, deep focus, volumetric light shafts, desaturated industrial greens against warm amber highlights, Kodak 2383 print stock, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 404

# 5. MACRO ALCHEMY — Extreme close-up chemical reactions on glass
progress "Macro Alchemy"
run_render "05_macro_alchemy" \
  --prompt "An extreme close-up of luminous liquid spreading across a sheet of black glass in a dark laboratory. Deep indigo ink meets a pool of molten copper-colored fluid, and where they touch a fractal boundary forms, branching into thousands of microscopic dendrites that crawl outward like frozen lightning. Tiny bubbles rise through the mixture catching overhead light as perfect spheres of white. The camera pushes in slowly, revealing ever finer detail in the crystalline reaction front, each branch splitting into smaller branches in an endless cascade. A single overhead spotlight creates a hard circular reflection on the glass surface. Macro lens at f/2.8, extreme shallow depth of field, rich saturated jewel tones against absolute black, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 505

# 6. SNOW CABIN — Isolation, warmth against cold
progress "Snow Cabin"
run_render "06_snow_cabin" \
  --prompt "A wide establishing shot of a remote wooden cabin during a heavy snowstorm at night, viewed from a distance. Warm firelight flickers through two small frost-edged windows, casting amber rectangles onto the snow-covered ground outside. Thick snowflakes fall steadily through the frame, each flake visible and sharp against the warm glow of the windows. Chimney smoke rises in a pale column before dissolving into the dark sky. Snow accumulates on the peaked roof and weighs down the branches of nearby pines. The camera pushes in slowly across the snow-covered clearing toward the cabin, the warm interior glow growing brighter. 50mm f/1.8, shallow depth of field, extreme contrast between blue-white exterior cold and deep amber interior warmth, Kodak Vision3 500T, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 606

# 7. DESERT HIGHWAY — American Southwest, blue hour loneliness
progress "Desert Highway"
run_render "07_desert_highway" \
  --prompt "A wide establishing shot of an empty two-lane highway cutting straight through the Utah desert at blue hour, the sky a deep cobalt gradient fading to burnt orange at the horizon. A single pair of headlights approaches from the vanishing point, growing slowly brighter against the vast emptiness. Red sandstone mesas rise on either side like ancient sentinels, their surfaces catching the last violet light of dusk on their western faces. The road surface is cracked and sun-bleached, center line paint faded to nothing. The camera holds steady on a low tripod inches above the asphalt, the road stretching out in perfect one-point perspective. 35mm f/4, deep focus, desaturated earth tones against electric blue sky, Kodak Ektachrome transparency film look, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 707

# 8. UNDERWATER CATHEDRAL — Sunken architecture, god rays
progress "Underwater Cathedral"
run_render "08_underwater_cathedral" \
  --prompt "A slow upward tilt inside a sunken stone cathedral submerged in crystal-clear tropical water. Massive columns draped in sea fans and soft coral rise from a sandy floor into a vaulted ceiling where shafts of sunlight penetrate through collapsed sections, each beam a solid column of turquoise light filled with drifting particles. Schools of small silver fish move in synchronized clouds between the columns, catching the light as brief flashes of white. Anemones in vivid orange and purple cling to carved stone arches, their tentacles swaying gently in an invisible current. The camera tilts slowly upward from floor level, revealing the full scale of the flooded nave. 16mm wide angle, deep focus, volumetric god rays, saturated tropical color palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up \
  --lora-weight-name ltx-2-19b-lora-camera-control-jib-up.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 808

# 9. TOKYO TRAIN — Late night commuter, fluorescent melancholy
progress "Tokyo Train"
run_render "09_tokyo_train" \
  --prompt "A medium close-up of a young man asleep on an empty late-night Tokyo commuter train, his head resting against the window as the city streaks past outside in long horizontal smears of white and amber light. His reflection ghosts faintly in the glass, overlapping with the rushing cityscape beyond. Overhead fluorescent tubes cast a flat greenish-white light that hollows his eye sockets and sharpens his cheekbones. A single forgotten umbrella rests in the seat beside him. The train sways gently, his body rocking with the rhythm of the rails. The camera holds still, tripod-locked to the train car, framing him against the streaming window. 50mm f/2, shallow depth of field, cold fluorescent interior against warm exterior motion blur, Fujifilm Eterna 500 film stock, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 909

# 10. FORGE — Blacksmith at work, fire and metal
progress "Forge"
run_render "10_forge" \
  --prompt "A slow dolly-left reveals a blacksmith working at a glowing forge in a dark stone workshop. His muscular forearms gleam with sweat as he brings a bar of white-hot steel down on the anvil, sending a shower of bright orange sparks cascading in graceful arcs that bounce off the stone floor. The forge behind him breathes with deep red and orange light, illuminating racks of hand-forged tools hanging on the soot-blackened walls. His leather apron catches the shifting warm tones of the fire. The camera drifts slowly left, revealing the full depth of the workshop as shadows dance with each hammer strike. 35mm f/2.8, shallow depth of field, extreme chiaroscuro lighting, rich warm palette of orange and deep black, Kodak 2383 print stock, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-left.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 1010

# 11. BIOLUMINESCENT TIDE — Night beach, living ocean
progress "Bioluminescent Tide"
run_render "11_bioluminescent_tide" \
  --prompt "A wide shot of a dark tropical beach at midnight where every breaking wave erupts in vivid electric blue bioluminescence, the foam tracing the entire shoreline in pulsing neon. Wet sand glows where the water retreats, the luminescence clinging to every surface it touches. The Milky Way stretches overhead in a dense band of purple and white, reflecting faintly in the thin sheet of water covering the tidal flat. Dark volcanic rocks in the foreground are outlined in blue where waves lap against them. The camera tracks slowly right along the shoreline, paralleling the glowing surf. 16mm wide angle, deep focus, extreme contrast between electric blue bioluminescence and the dark starlit sky, 180-degree shutter, natural motion blur, raw footage aesthetic." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 1111

# 12. LIBRARY DUST — Ancient knowledge, volumetric cathedral light
progress "Library Dust"
run_render "12_library_dust" \
  --prompt "A slow jib-down descends from the vaulted ceiling of an enormous baroque library, revealing tier upon tier of leather-bound books stretching from floor to ceiling on dark mahogany shelves. Three tall arched windows pour thick shafts of golden morning light through dust-filled air, each beam so dense with particles it looks solid enough to touch. A massive oak reading table sits below, covered in open manuscripts and brass instruments. The camera lowers smoothly from ceiling height to eye level, the scale of the room becoming overwhelming as shelves tower overhead into shadow. Motes of dust spiral lazily through the light beams in slow hypnotic patterns. 35mm f/2.8, deep focus, volumetric god rays, warm amber and deep brown palette, Kodak Portra film stock, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down \
  --lora-weight-name ltx-2-19b-lora-camera-control-jib-down.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 1212

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo ""
echo "==========================================" | tee -a "$LOG"
echo "  Batch 1 complete — 12 renders" | tee -a "$LOG"
echo "  Total time: ${TOTAL_MIN} minutes" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "  Each render: .mp4 + _first.png + _last.png" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
