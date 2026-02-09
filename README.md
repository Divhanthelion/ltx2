# LTX-2 Video Generation on NVIDIA Jetson Thor

> **This project is built specifically for the [NVIDIA Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) (128 GB unified CPU/GPU memory).** The memory lifecycle, VAE decode strategy, and offload hook workarounds are all tuned for Jetson Thor's unified memory architecture. It may work on other 128 GB+ NVIDIA GPU systems, but it has only been tested on the Thor.

A production pipeline for generating high-quality AI video with [LTX-2](https://huggingface.co/Lightricks/LTX-2) on Jetson AGX Thor. Runs the 19B-parameter distilled model with camera control LoRAs, with memory lifecycle management tuned for unified memory (no CPU offload — everything loads directly onto CUDA).

## What this does

- Generates 1080p video (1920x1088) at 24fps with synchronized audio (LTX-2 generates both)
- 8-step distilled inference (~15 min diffusion + ~2 min decode per clip)
- Camera control via 7 official LoRAs (dolly, jib, static)
- Optional FP8 quantization for the transformer (saves ~10 GB for memory-constrained systems)
- Single-pass VAE decode with streaming to disk (optional chunking for lower-memory systems)
- Automatic latent backup for crash recovery
- Batch rendering scripts with progress tracking and ETA

## Hardware Requirements

**Built for and tested on NVIDIA Jetson AGX Thor** (Blackwell GPU, 128GB unified memory, JetPack 7.0, CUDA 13.0, sm_110).

The 19B-parameter model plus VAE decode at 1080p requires **~80-100 GB of memory** at peak. The pipeline is designed around the Jetson Thor's 128GB unified memory architecture, which requires specific memory management patterns — see [Memory Management](#memory-management-on-unified-memory) below.

**Minimum: 128 GB unified or VRAM.** If you have less than this (e.g. 64 GB), expect the Linux OOM killer to terminate the process during VAE decode. You can work around this with `--vae-chunk-frames 4` (see [VAE chunking](#vae-decode-and-chunking)), but this introduces visible temporal stitch artifacts. You could also generate at lower resolutions (768x512) where memory pressure is much lower.

**Do not run this on a 64 GB system without chunking.** Without `--vae-chunk-frames`, the single-pass VAE decode will exhaust memory and the kernel will kill the process with no warning — you'll just see the container disappear. Your latents are auto-saved so you can recover, but it will waste ~15 minutes of diffusion time per failed attempt.

## Quick Start

### 0. Prerequisites

- **NVIDIA Jetson AGX Thor** with JetPack 7.0 (CUDA 13.0, sm_110)
- **Docker** with `nvidia-container-runtime` (ships with JetPack)
- **~80 GB free disk** for the model cache (~38 GB model + LoRAs + Docker image)
- **sudo access** for Docker and kernel cache flushing

### 1. Build the container

```bash
docker build -t ltx2 .
```

The Dockerfile is based on [`nvcr.io/nvidia/pytorch:26.01-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) (NGC PyTorch for JetPack 7.0). Key details:

- **Do not reinstall PyTorch** — the NGC image includes PyTorch built for CUDA 13.0 + sm_110 (Blackwell). pip-installing a different PyTorch will break GPU support.
- **NGC pip constraints are neutralized** — the base image ships with pip constraint files that block newer package versions. The Dockerfile removes these so diffusers can install from git main.
- **Diffusers is installed from git main** — the released PyPI version may not include `LTX2Pipeline`. The git main branch is required.
- **`HF_HUB_OFFLINE=1`** — the container runs in offline mode. All models must be downloaded to your host's HuggingFace cache before running (see step 2).

### 2. Download the model (first run only)

This pipeline defaults to [`rootonchair/LTX-2-19b-distilled`](https://huggingface.co/rootonchair/LTX-2-19b-distilled), a community-hosted distilled version of LTX-2 that runs in 8 inference steps (~15 min on Jetson Thor). **You must download it before your first run** — the container mounts your local HuggingFace cache and does not download models at runtime.

```bash
# Download the distilled model (~38 GB)
huggingface-cli download rootonchair/LTX-2-19b-distilled

# Download camera control LoRAs (optional, ~200 MB each)
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Static
```

**Using the official full model instead:** If you prefer the non-distilled model from Lightricks, pass `--model Lightricks/LTX-2` and change the inference settings to `--guidance-scale 3.5 --steps 40`. The full model is ~5x slower but supports negative prompts via CFG.

### 3. Generate a video

```bash
sudo docker run --rm --runtime=nvidia --privileged \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/ltx2/outputs:/outputs ltx2 \
  --prompt "A slow dolly-in through a rain-soaked alleyway in Taipei at 2 AM. Neon signs reflect off wet cobblestones in smeared reds and electric blues. Steam rises from a vendor cart, catching the colored light. 35mm f/2.8, shallow depth of field, Kodak Vision3 500T, 180-degree shutter, natural motion blur." \
  --lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In \
  --lora-weight-name ltx-2-19b-lora-camera-control-dolly-in.safetensors \
  --width 1920 --height 1088 --num-frames 161 --no-fp8 --seed 101
```

`--privileged` is required for flushing the kernel page cache (`/proc/sys/vm/drop_caches`) after all models are freed and the video is written. Without it, cache flushing silently fails — generation still works but memory reclamation is less effective between batch runs.

Output: `/outputs/video_YYYYMMDD_HHMMSS.mp4` (timestamp auto-appended to prevent overwrites)

### 4. Recover from a crashed decode

If the VAE decode OOMs, latents are auto-saved. Decode them later:

```bash
sudo docker run --rm --runtime=nvidia --privileged \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/ltx2/outputs:/outputs ltx2 \
  decode --input /outputs/video_latents.npz --output /outputs/recovered.mp4
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | (required) | Plain natural language prompt paragraph |
| `--negative-prompt` | `""` | Negative prompt (see [note on negative prompts](#negative-prompts-do-not-work-with-the-distilled-model)) |
| `--model` | `rootonchair/LTX-2-19b-distilled` | HuggingFace model ID. For the full (non-distilled) model, use `Lightricks/LTX-2` with `--guidance-scale 3.5 --steps 40` |
| `--output` | `/outputs/video.mp4` | Output path (timestamp auto-appended) |
| `--width` | `512` | Width (must be divisible by 32) |
| `--height` | `768` | Height (must be divisible by 32) |
| `--num-frames` | `97` | Frame count (must be `8n+1`: 9, 17, 25... 97, 121, 161, 257) |
| `--steps` | `8` | Inference steps (8 for distilled model) |
| `--guidance-scale` | `1.0` | CFG scale (1.0 for distilled model) |
| `--seed` | random | Reproducible seed |
| `--no-fp8` | off | Disable FP8 quantization (recommended on 128 GB systems for best quality) |
| `--lora` | none | HuggingFace LoRA repo ID |
| `--lora-weight-name` | none | Specific `.safetensors` file in the LoRA repo |
| `--lora-scale` | `1.0` | LoRA adapter strength (0.0-1.0) |
| `--vae-chunk-frames` | auto | Override VAE temporal chunk size (see [VAE chunking](#vae-decode-and-chunking)) |
| `--sequential-offload` | off | Per-layer CPU offload (slower, lower peak memory) |

## Optimal Settings

For Jetson Thor with 128GB unified memory:

| Setting | Value | Why |
|---------|-------|-----|
| Resolution | **1920x1088** | Model trained at 1080p. Height 1088 = divisible by 32. |
| Frames | **161** | `8n+1` rule. ~6.7s at 24fps. Good quality/memory balance. |
| Steps | **8** | Distilled model schedule |
| Guidance scale | **1.0** | Distilled model (CFG disabled) |
| FPS | **24** | Cinematic standard, matches training data |

**Frame count must follow the `8n+1` rule**: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, ..., 257.

For higher resolution output, generate at 1080p and upscale externally with ESRGAN, Topaz, or the LTX Video Spatial Upscaler. Direct generation above 1080p produces spatial tiling artifacts — see [Known Issues](#known-issues).

## Technical Findings

### Negative prompts do not work with the distilled model

The distilled model uses `guidance_scale=1.0`. Classifier-Free Guidance computes:

```
output = unconditional + guidance_scale * (conditional - unconditional)
```

At `guidance_scale=1.0`, this simplifies to `output = conditional`. The negative prompt embedding is computed but **mathematically eliminated** — it has zero effect on the output. The `--negative-prompt` flag is accepted silently but does nothing.

**Options if you need negative prompts:**
- Switch to the full model (`Lightricks/LTX-2`) with `--guidance-scale 3.5 --steps 40` (~5x slower)
- Implement [Normalized Attention Guidance (NAG)](https://chendaryen.github.io/NAG.github.io/) which works in attention space and functions at CFG=1.0

### VAE decode and chunking

By default, `generate.py` decodes latents through the VAE in a **single pass**. This produces the cleanest output — the VAE's built-in temporal tiling (16-frame tiles, stride 8) handles any necessary internal splitting with proper coherence.

Single-pass decode at 1920x1088 with 161 frames requires **~60-80 GB** for VAE intermediates. This fits comfortably on 128 GB unified memory after the transformer and text encoder are freed (see [Memory Management](#memory-management-on-unified-memory)).

**Why not chunk by default?** Manual temporal chunking splits the latent tensor into pieces, decodes each independently, then blends overlapping regions with linear crossfade. Adjacent chunks disagree on what the overlap frames look like, producing visible **doubling/stutter** artifacts at each stitch boundary. Frame-difference analysis on a 1920x1088@161 test showed the highest frame-to-frame pixel deltas concentrated exactly in the blend zones.

**If you're running on a system with less than 128 GB**, you can enable manual chunking with `--vae-chunk-frames`:

```bash
# Fewer chunks = fewer stitch artifacts, but more memory per chunk
--vae-chunk-frames 11   # 2 chunks for 161 frames, 1 stitch point (~40 GB)
--vae-chunk-frames 8    # 3 chunks for 161 frames, 2 stitch points (~30 GB)
--vae-chunk-frames 4    # 6 chunks for 161 frames, 5 stitch points (~15 GB)
```

The chunk size refers to latent temporal frames (each latent frame = 8 pixel frames). Lower values use less memory but introduce more stitch artifacts.

### VAE spatial tiling at high resolution

The VAE's `enable_tiling()` uses 512x512 spatial tiles with 64px overlap (stride 448). At 1920x1088, the entire frame fits comfortably and tiling artifacts are not visible. At 2560x1440+, the 64px overlap produces visible horizontal/vertical banding seams.

If you need to experiment with high-res tiling:

```python
vae.enable_tiling(
    tile_sample_min_height=1024,
    tile_sample_min_width=1024,
    tile_sample_stride_height=896,   # overlap = 128px
    tile_sample_stride_width=896,
)
```

Community consensus: generate at 1080p and upscale. The model was trained at 1080p and produces significantly better results within that distribution.

### Memory management on unified memory

Jetson Thor's CPU and GPU share the same 128GB RAM pool. This changes how memory management works:

- **`tensor.to("cpu")` does nothing** — CPU and GPU are the same physical memory. You must `del` the object.
- **Do NOT use `enable_model_cpu_offload()`** — on unified memory, CPU offload is pointless (`.to("cpu")` doesn't free anything) and actively harmful. The offload hooks create hidden reference paths that prevent model deletion, and removing them later leaves models in an inconsistent device state that causes segfaults during VAE decode. Instead, `generate.py` loads the full pipeline onto CUDA with `pipe.to(device)`.
- **Python GC alone is too slow** — explicitly `del` every large object when done, then `torch.cuda.empty_cache()` + `gc.collect()` twice (second pass catches objects freed by first).
- **Flush kernel page cache only AFTER the video is written** — `echo 3 > /proc/sys/vm/drop_caches` reclaims mmap'd safetensors pages. But on unified memory, model weights may still be backed by these pages even after `.to("cuda")`. Dropping the cache while models are alive evicts their weights and causes segfaults. Only drop caches after all models are deleted and the video is on disk.
- **Save metadata before deleting models** — e.g., grab `vocoder.config.output_sampling_rate` before deleting the vocoder.

The pipeline lifecycle in `generate.py`:

1. Load full pipeline onto CUDA, run diffusion, extract latents, **delete transformer + text_encoder + tokenizer + scheduler**
2. Decode audio latents, **delete audio_vae + vocoder + audio_projection**
3. Decode video latents through VAE (single-pass, streaming to disk), **delete entire pipeline, flush page cache**
4. Encode video with ffmpeg from raw frames on disk

## Camera Control LoRAs

All LoRAs are from `Lightricks/LTX-2-19b-LoRA-Camera-Control-*`:

| LoRA | Weight file | Effect |
|------|-------------|--------|
| Dolly-In | `ltx-2-19b-lora-camera-control-dolly-in.safetensors` | Push forward |
| Dolly-Out | `ltx-2-19b-lora-camera-control-dolly-out.safetensors` | Pull back |
| Dolly-Left | `ltx-2-19b-lora-camera-control-dolly-left.safetensors` | Track left |
| Dolly-Right | `ltx-2-19b-lora-camera-control-dolly-right.safetensors` | Track right |
| Jib-Up | `ltx-2-19b-lora-camera-control-jib-up.safetensors` | Crane up |
| Jib-Down | `ltx-2-19b-lora-camera-control-jib-down.safetensors` | Crane down |
| Static | `ltx-2-19b-lora-camera-control-static.safetensors` | Tripod lock |

**Load order matters**: base model, then LoRA, then FP8 casting. Reversing LoRA and FP8 breaks the adapter weights (see [diffusers #11648](https://github.com/huggingface/diffusers/issues/11648)).

## Batch Scripts

Pre-built batch scripts for themed render sessions, each generating 12-16 clips at 1920x1088@161 frames with camera LoRAs:

| Script | Theme | Clips |
|--------|-------|-------|
| `batch1.sh` | Cinematic scenes (noir, deep sea, forge, etc.) | 12 |
| `ltx2_vivid_batch2.sh` | Environments (Venice fog, jungle, volcanic glass, etc.) | 12 |
| `ltx2_batch3_robots.sh` | Near-future robotics (Boston Dynamics style) | 12 |
| `ltx2_batch4_puppies.sh` | Puppy chaos (toilet paper, mud, zoomies, etc.) | 12 |
| `ltx2_batch5_archaeology.sh` | Ancient sites (Stonehenge, Petra, Angkor Wat, etc.) | 16 |
| `test_all_loras.sh` | Same prompt with all 7 LoRAs for comparison | 7 |

Each script includes progress tracking with ETA, first/last frame extraction, cache flushing between renders, and automatic cleanup.

## Roadmap

- **Image-to-video generation** — LTX-2 supports conditioning on a reference image (`image=` parameter in `LTX2Pipeline`), but this pipeline currently only supports text-to-video. An `--image` flag is planned.

## Known Issues

- **Frame count mismatch**: Some renders produce slightly fewer frames than requested (e.g., 148 instead of 161). Likely a VAE temporal boundary rounding edge case.
- **Spatial tiling seams at 2560x1440+**: Visible horizontal banding from default 64px tile overlap. Stay at 1080p.
- **Scene coherence loss on long sequences**: At 321+ frames the model can lose narrative coherence in the second half, especially at higher resolutions.
- **Content duplication at high res**: At 2560x1440, top/bottom halves of the frame can show mirrored content from spatial tiling.

## Project Structure

```
generate.py            Main pipeline — diffusion, audio decode, VAE decode, export
decode_latents.py      Standalone latent-to-video decoder for crash recovery
entrypoint.sh          Docker entrypoint
Dockerfile             Container build (NGC PyTorch 26.01 base)
PROMPTING_GUIDE.md     Detailed prompting strategies for LTX-2
batch1.sh              Batch render scripts (see table above)
ltx2_vivid_batch2.sh
ltx2_batch3_robots.sh
ltx2_batch4_puppies.sh
ltx2_batch5_archaeology.sh
test_all_loras.sh
```

## References

- [LTX-2 Model Card](https://huggingface.co/Lightricks/LTX-2)
- [LTX-2 Distilled](https://huggingface.co/rootonchair/LTX-2-19b-distilled)
- [Diffusers LTX2Pipeline API](https://huggingface.co/docs/diffusers/main/api/pipelines/ltx2)
- [AutoencoderKLLTX2Video API](https://huggingface.co/docs/diffusers/main/api/models/autoencoderkl_ltx_2)
- [Camera Control LoRAs](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In)
- [NAG (Normalized Attention Guidance)](https://chendaryen.github.io/NAG.github.io/)
