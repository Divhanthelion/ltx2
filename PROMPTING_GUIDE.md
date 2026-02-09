# LTX-2 Prompting Guide

## The Golden Rule

**If a real camera crew could execute the shot without asking follow-up questions, LTX-2 will usually deliver.**

LTX-2 wants a physically plausible scene that unfolds over time. If the shot would be impossible, contradictory, or vague, the model compensates by inventing motion, morphing objects, or multiplying elements.

## Prompt Format

- Write a **single flowing paragraph** — no bullet lists, no line breaks
- Use **present-tense verbs** exclusively ("walks", "tilts", "flickers" — not "walked")
- Aim for **4-8 sentences**, roughly **200 words max**
- Use temporal connectors: "as", "then", "while", "before", "after"
- **NO tag-style prompting** — comma-separated keyword lists (Stable Diffusion style) produce poor results

## Prompt Element Order

LTX-2 weights the **beginning of the prompt more heavily**. Put the most important elements first.

**Recommended order:**

1. **Shot type** — "wide establishing shot", "extreme close-up", "low-angle"
2. **Subject & action** — present tense, concrete physical descriptions
3. **Environment** — setting, atmosphere, background details
4. **Camera movement** — one clean move per shot
5. **Lighting & color** — motivated light sources, color palette
6. **Lens & film stock** — "50mm", "Kodak 2383", "shallow depth of field"

### Template

> "A [shot size] of [subject] performing [continuous action]. The scene takes place in [environment]. The camera [specific movement]. Lighting is [description]. Shot on [lens/film stock]."

## Duration vs Complexity

| Duration | What fits |
|----------|-----------|
| Under 5s (~97 frames) | Single action, simple camera move |
| 5-10s (~121-241 frames) | 2-3 connected actions, one camera move |
| 10+ seconds | Multiple sequences, camera changes, environmental shifts |

Too much action crammed into a short duration = rushed, incoherent motion. Too little = drift and morphing.

## What Works Well

- **Cinematic compositions** with thoughtful lighting and shallow DoF
- **Emotive human moments** — subtle gestures, facial nuance, body language
- **Atmospheric details** — fog, mist, golden hour, soft shadows, rain, reflections
- **Clean camera language** — "slow dolly", "handheld tracking", "tripod-locked"
- **Stylized aesthetics** — painterly, noir, analog film, fashion editorial
- **Dancing** — one of the few dynamic movements that works reliably
- **Characters speaking** in various languages
- **Single, clean camera movements** per shot

## What Does NOT Work

- **Text/logos** — cannot generate readable text of any kind
- **Complex physics** — smoke, fire particles, liquid simulations, collisions, juggling
- **Object multiplication** — repeated similar elements (pipes, fingers, limbs) tend to multiply
- **Too many characters** — multi-person interactions produce artifacts
- **Abstract emotion labels** — "he felt sad" does nothing; "his shoulders drop" works
- **Vague vibe words** — "cozy minimalist" loses to specific physical descriptions
- **Multiple camera movements** — causes jitter and temporal wobble; pick ONE per shot
- **Strong handheld** — causes edge distortion; use "subtle handheld, micro-shakes only"

## Camera Movements (Tested Results)

| Movement | Quality |
|----------|---------|
| Dolly-in/out | Smooth, minimal jitter |
| Slow pan left/right | Gentle, straight horizons |
| Track forward/back | Clean following motion |
| Subtle tilt up/down | Good for reveals |
| Static lock-off | Crisp detail |
| Digital zoom | Hunts focus — dolly-in is better |
| Strong handheld | Edge distortion — avoid |

## Using Camera Control LoRAs

Available LoRAs: `Dolly-In`, `Dolly-Out`, `Dolly-Left`, `Dolly-Right`, `Jib-Up`, `Jib-Down`, `Static`

All under `Lightricks/LTX-2-19b-LoRA-Camera-Control-*`

**Key tip:** When using motion LoRAs, **describe what the camera reveals** as it moves. For a dolly-right, describe what comes into view on the right. Give the model something to show.

### LoRA Scale

| Scale | Effect |
|-------|--------|
| 0.9-1.1 | Subtle, preserves base style |
| 1.2-1.4 | Balanced (start here) |
| 1.5-1.6 | Maximum intensity |

## Lens Specs That Help

Adding a lens specification reduces edge shimmer and improves coherence:

- **35mm f/2.8** — neutral cinematic default, people/products
- **50mm** — intimate, human scale
- **85mm** — talking heads, portrait bokeh
- **16mm wide** — rooms, deep focus
- **Macro** — extreme detail work

## Stability & Realism Anchors

Pick one from each category and include in your prompt:

**Stability:** "tripod-locked", "stabilized footage", "slow smooth motion"

**Realism:** "live action", "raw footage", "4K", "high fidelity"

**Style (pick ONE):** "Kodak Portra", "chiaroscuro", "volumetric fog", "desaturated cinematic"

Mixing multiple style anchors leads to visual mud.

## Motion Realism

Adding **"180-degree shutter"** or **"natural motion blur"** reduces stutter and gives motion physical weight. This is one of the few technical phrases that reliably improves output.

## Negative Prompts

**The distilled model ignores negative prompts entirely.** At `guidance_scale=1.0`, the CFG math simplifies to `output = conditional` — the negative prompt embedding is computed but mathematically eliminated. The `--negative-prompt` flag is accepted silently but has zero effect.

If you need negative prompt control, switch to the full model (`Lightricks/LTX-2`) with `--guidance-scale 3.5 --steps 40`. With the full model, keep negatives short and targeted (5-8 items max):

```
morphing, distortion, warping, flicker, jitter, temporal artifacts,
low quality, text, watermark, logo, cartoon, anime, CGI
```

For hands/faces add: `deformed hands, extra fingers, malformed face`

## Resolution & Settings Reference

| Setting | Default | Notes |
|---------|---------|-------|
| Resolution | 1920x1088 | Model trained at 1080p. Must be divisible by 32. |
| Frames | 161 (~6.7s) | Must be 8n+1 (9, 17, 25... 97, 121, 161... 257) |
| Steps | 8 | For distilled model |
| Guidance scale | 1.0 | For distilled model |
| FPS | 24 | Cinematic standard |

**Resolution sweet spot:** 1920x1088 for production quality. 768x512 for fast iteration. For 4K output, generate at 1080p and upscale with ESRGAN/Topaz. Direct generation above 1080p causes spatial tiling artifacts.

## Example Prompts

### Simple (good for testing)

> A slow dolly-in on a weathered leather journal lying open on a dark oak desk. Warm amber light from a nearby lamp illuminates handwritten pages as dust motes drift through the beam. The camera pushes forward smoothly, revealing ink illustrations in the margins. 50mm lens, shallow depth of field, Kodak Portra palette, natural motion blur.

### Character-focused

> A woman in her thirties stands at the edge of a rooftop at dusk, wind catching her dark hair across her face. She turns slowly toward the camera, her expression shifting from distant thought to a faint, knowing smile. The city skyline glows orange and purple behind her. The camera holds steady on a tripod, framing her from the waist up. Golden hour backlight, 85mm portrait lens, shallow depth of field, rich saturated color.

### Environment-heavy

> A slow tracking shot moves through an abandoned greenhouse overtaken by tropical plants. Shattered glass panels let columns of golden afternoon light pour across tangled vines and moss-covered stone paths. Water drips steadily from rusted iron beams into shallow pools below. The camera glides forward at waist height, stabilized and smooth. 35mm lens, deep focus, volumetric light shafts, desaturated greens and warm amber highlights.
