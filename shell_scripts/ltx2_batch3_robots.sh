#!/bin/bash
# LTX-2 Batch 3 — ROBOTS — 1920x1088 @ 161 frames (~6.7s)
# Near-future realistic, Boston Dynamics style
# NOTE: no set -e — we want the batch to continue even if one render fails

TOTAL=12
CURRENT=0
START_TIME=$(date +%s)
OUTPUT_DIR="$(pwd)/outputs"
BATCH_DIR="$OUTPUT_DIR/batch3_robots"
LOG="$BATCH_DIR/batch3_log.txt"

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
echo "  LTX-2 Batch 3 — ROBOTS — $(date)" | tee -a "$LOG"
echo "  Resolution: 1920x1088 | Frames: 161" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# 1. QUADRUPED RAIN — Boston Dynamics-style dog robot in a downpour
progress "Quadruped Rain"
run_render "01_quadruped_rain" \
  --prompt "A medium shot of a matte black quadruped robot walking steadily through a heavy rainstorm on a wet concrete loading dock. Its four articulated legs move with precise mechanical rhythm, each carbon-fiber joint bending and extending with hydraulic smoothness as rubber-padded feet find grip on the slick surface. Water streams down its angular chassis, pooling in the recessed sensor housings and dripping from cable conduits along its underside. Mounted lidar units on its back spin continuously, red laser lines faintly visible through the rain. The camera tracks alongside at the robot's eye level, matching its pace. Overcast daylight, wet industrial grey palette, every surface glistening, 35mm f/2.8, shallow depth of field, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3001

# 2. HUMANOID WAREHOUSE — Bipedal robot reaching for a shelf
progress "Humanoid Warehouse"
run_render "02_humanoid_warehouse" \
  --prompt "A slow dolly-in on a tall bipedal humanoid robot standing in the aisle of a cavernous fulfillment warehouse, its white polymer shell panels catching the flat overhead LED light. The robot reaches forward with a five-fingered gripper hand, the silicone-tipped fingers closing precisely around a cardboard box on a high shelf. Visible actuator cables run along its forearms like mechanical tendons, tightening as the grip engages. Its head is a smooth featureless ovoid with a horizontal strip of blue sensor LEDs that pulse softly. Behind it, warehouse shelving stretches into deep perspective under harsh industrial fluorescents. The camera pushes in slowly from waist level. 50mm f/2, shallow depth of field, clinical white and industrial grey palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3002

# 3. SEARCH AND RESCUE — Tracked robot in rubble
progress "Search Rescue"
run_render "03_search_rescue" \
  --prompt "A close-up tracking shot follows a compact treaded search-and-rescue robot as it crawls over a pile of concrete rubble and twisted rebar in a collapsed building. Its rubberized tank treads grip and flex over jagged debris, the chassis tilting and self-leveling with each obstacle. An articulated sensor mast extends from its top, a thermal camera and LED spotlight swiveling as they scan the darkness ahead. Orange emergency strobes on its flanks pulse rhythmically, casting harsh orange flashes across the grey dust-covered wreckage. Fine concrete dust hangs in the air, catching the spotlight beam. The camera follows low to the ground, just behind the robot. 35mm f/2.8, shallow depth of field, emergency orange against concrete grey, volumetric dust, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3003

# 4. ASSEMBLY LINE — Robotic arms in synchronized welding
progress "Assembly Line"
run_render "04_assembly_line" \
  --prompt "A wide dolly-right along a modern automotive assembly line where six identical orange industrial robotic arms perform synchronized spot-welding on a bare car chassis. Each arm moves with fluid six-axis articulation, the welding tips producing brief intense white sparks that scatter in parabolic arcs and reflect off the polished concrete floor. Cable looms snake from each arm's base in neat bundles, and safety-yellow bollards mark the perimeter of the work cell. The camera glides smoothly past the row of robots, each one entering the frame performing a different phase of its routine. Cool blue-white factory lighting from above, warm orange sparks providing counterpoint. 35mm f/4, deep focus, industrial orange and safety yellow against steel grey, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3004

# 5. DRONE SWARM — Quadcopters lifting off in formation over farmland
progress "Drone Swarm"
run_render "05_drone_swarm" \
  --prompt "A wide establishing shot looking across flat midwestern farmland at golden hour as a grid of twelve agricultural drones lifts off simultaneously from a launch pad, each quadcopter rising on four blurred rotor discs that kick up a thin layer of dust from the dry soil. The drones ascend in perfect formation, maintaining equal spacing as they climb, their white shells catching the low amber sunlight while their undersides fall into cool shadow. Green and red navigation LEDs blink in synchronized patterns on each unit. Below them, rows of young corn stretch to the horizon. The camera holds steady on a tripod, the formation rising through the frame against a vast prairie sky. 50mm f/4, deep focus, warm golden light on white drones against blue sky, Kodak Portra palette, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3005

# 6. LAB HAND — Robotic hand picking up an egg
progress "Lab Hand"
run_render "06_lab_hand" \
  --prompt "An extreme close-up of an advanced robotic hand picking up a raw chicken egg from a foam holder on a lab bench under a bright ring light. The hand has five articulated fingers with transparent silicone fingertip pads, revealing pressure sensor arrays embedded inside as grids of tiny gold contact points. The fingers curve around the shell with sub-millimeter precision, visible force-feedback cables along each finger adjusting tension in real time. The egg lifts smoothly without a crack. Machined aluminum knuckle joints gleam under the clinical white light, and thin ribbon cables run from each fingertip back along the wrist housing. The camera pushes in slowly. Macro lens at f/2.8, extreme shallow depth of field, clinical white and brushed aluminum palette, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3006

# 7. SNOW PATROL — Quadruped robot trotting through deep snow
progress "Snow Patrol"
run_render "07_snow_patrol" \
  --prompt "A wide tracking shot of a bright yellow quadruped robot trotting through knee-deep fresh snow in a dense pine forest at dusk. Each leg lifts high and plunges back through the powder, sending small puffs of snow crystals into the blue twilight air. Its heated sensor dome on top emits a faint column of steam in the cold. LED headlights mounted on its chest throw two sharp white beams forward through the falling snow, each flake illuminated as a brief bright streak. Pine branches sag heavy with snow on either side, forming a natural corridor. The camera tracks alongside at a moderate distance. Blue hour light, cold blue-white palette with the warm yellow chassis as focal point, 35mm f/2.8, shallow depth of field, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-left.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3007

# 8. CHARGING DOCK — Robot powering down in maintenance bay
progress "Charging Dock"
run_render "08_charging_dock" \
  --prompt "A slow dolly-in on a humanoid robot settling into its charging dock in a quiet maintenance bay late at night. The robot backs into the cradle, alignment pins engaging with a subtle mechanical click as charging contacts connect at its lower spine. Its status LEDs transition from active blue to a slow-breathing amber pulse as it enters standby mode. The head tilts forward slightly, chin dropping, in an uncanny echo of a human falling asleep. A single desk lamp illuminates the scene from the left, casting long warm shadows across tool racks and dark diagnostic monitors. The camera pushes in slowly, framing the robot from the chest up as its systems wind down. 85mm f/1.8, shallow depth of field, warm amber desk lamp against cool blue standby LEDs, Kodak Vision3 500T, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3008

# 9. UNDERWATER ROV — Deep sea hull inspection
progress "Underwater ROV"
run_render "09_underwater_rov" \
  --prompt "A medium tracking shot follows a bright orange underwater ROV as it glides along the hull of a massive cargo ship in murky green harbor water. The ROV's four vectored thrusters adjust continuously, tiny jets of bubbles streaming from each nozzle as it maintains precise station-keeping. A pair of high-intensity LED panels on its front illuminate a section of the barnacle-crusted hull in stark white light, revealing layers of marine growth in greens and browns against the red anti-fouling paint beneath. A manipulator arm extends from beneath the ROV, its claw gently probing a corroded weld seam. Particles of marine debris drift through the light beams. The camera tracks alongside at arm's length. 35mm f/2.8, shallow depth of field, orange ROV against murky green water, 180-degree shutter, natural motion blur, stabilized footage." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-right.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3009

# 10. CONSTRUCTION SITE — Autonomous excavator digging at dawn
progress "Construction Site"
run_render "10_construction_site" \
  --prompt "A wide establishing shot of an autonomous excavator working alone on a muddy construction site in the first grey light of pre-dawn. The machine has no cab, its operator station replaced by a smooth sensor turret bristling with lidar pucks and stereo cameras that scan continuously. The bucket arm moves with eerie precision, scooping dark wet earth and depositing it in a neat pile, each motion repeating with mechanical exactness. Hazard beacons on the chassis rotate slowly, casting sweeping amber light across puddles and tire ruts. Fog sits low across the site, the machine emerging from it like something patient and autonomous. The camera holds on a distant tripod. 35mm f/4, deep focus, muted earth tones with amber hazard light, pre-dawn grey palette, Kodak Ektachrome, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3010

# 11. BIPEDAL STAIRS — Humanoid robot descending steps
progress "Bipedal Stairs"
run_render "11_bipedal_stairs" \
  --prompt "A low-angle shot looking up a concrete stairwell as a bipedal humanoid robot descends toward the camera with careful deliberate steps. Each foot placement is precise, the ankle actuators making micro-adjustments visible as small twitches in the joint housings. Its arms extend slightly for balance, fingertips spread, mimicking a human descending uncertain terrain. The robot's matte grey chassis is marked with research institution logos and serial numbers in small white text. Overhead fluorescent tubes cast flat greenish light down the stairwell, each step edge catching a sharp highlight. The camera holds low and still at the bottom of the stairs, the robot growing larger with each step. 35mm f/2.8, moderate depth of field, institutional grey and fluorescent green palette, 180-degree shutter, natural motion blur, tripod-locked." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3011

# 12. SUNSET SENTINEL — Quadruped silhouette on a ridge
progress "Sunset Sentinel"
run_render "12_sunset_sentinel" \
  --prompt "A wide cinematic shot of a quadruped patrol robot standing perfectly still on a rocky desert ridge at sunset, silhouetted against a sky of deep orange and magenta. Its angular chassis is outlined in golden rim light, every edge and antenna sharp against the burning horizon. A single red sensor eye on its head unit glows steadily, a tiny point of artificial light against the vast natural display behind it. Its four legs are planted wide in a stable stance on the uneven sandstone, cable housings and hydraulic lines visible along each limb. Far below, a desert valley stretches into hazy purple distance. The camera holds absolutely still, the only movement the slow shift of sunset color across the sky. 50mm f/4, deep focus, extreme silhouette contrast, warm sunset palette with single red sensor accent, Kodak Ektachrome, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
  --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 3012

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))

echo ""
echo "==========================================" | tee -a "$LOG"
echo "  Batch 3 ROBOTS complete — 12 renders" | tee -a "$LOG"
echo "  Total time: ${TOTAL_MIN} minutes" | tee -a "$LOG"
echo "  Output: $BATCH_DIR" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
