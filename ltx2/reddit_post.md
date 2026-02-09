# Running LTX-2 19B on a Jetson Thor — open-source pipeline with full memory lifecycle management

I've been running LTX-2 (the 19B distilled model) on an NVIDIA Jetson AGX Thor and built an open-source pipeline around it. Generating 1080p video (1920x1088) at 24fps with audio, camera control LoRAs, and batch rendering. Figured I'd share since there's almost nothing out there about running big video models on Jetson.

**GitHub: [LINK]**

## What it generates

[UPLOAD 2-3 SAMPLE CLIPS — noir_rain, deep_sea, and a puppy clip would be a good mix]

1920x1088, 161 frames (~6.7s), 24fps with synchronized audio. About 15 min diffusion + 2 min VAE decode per clip on the Thor.

## The interesting part: unified memory

The Jetson Thor has 128GB of RAM shared between CPU and GPU. This sounds great until you realize it breaks every standard memory optimization:

- **`enable_model_cpu_offload()` is useless** — CPU and GPU are the same memory. Moving tensors to CPU frees nothing. Worse, the offload hooks create reference paths that prevent model deletion, and removing them later leaves models in an inconsistent state that segfaults during VAE decode.
- **`tensor.to("cpu")` is a no-op** — same physical RAM. You have to actually `del` the object and run `gc.collect()` + `torch.cuda.empty_cache()` (twice — second pass catches objects freed by the first).
- **Page cache will kill you** — safetensors loads weights via mmap. Even after `.to("cuda")`, the original pages may still be backed by page cache. If you call `drop_caches` while models are alive, the kernel evicts the weight pages and your next forward pass segfaults.
- **You MUST use `torch.no_grad()` for VAE decode** — without it, PyTorch builds autograd graphs across all 15+ spatial tiles during tiled decode. On unified memory, this doesn't OOM cleanly — it segfaults. I lost about 4 hours to this one.

The pipeline does manual memory lifecycle: load everything → diffuse → delete transformer/text encoder/scheduler/connectors → decode audio → delete audio components → VAE decode under `no_grad()` → delete everything → flush page cache → encode video. Every stage has explicit cleanup and memory reporting.

## What's in the repo

- `generate.py` — the main pipeline with all the memory management
- `decode_latents.py` — standalone decoder for recovering from failed runs (latents are auto-saved)
- Batch rendering scripts with progress tracking and ETA
- Camera control LoRA support (dolly in/out/left/right, jib up/down, static)
- Optional FP8 quantization (cuts transformer memory roughly in half)
- Post-processing pipeline for RIFE frame interpolation + Real-ESRGAN upscaling (also Dockerized)

Everything runs in Docker so you don't touch your system Python. The NGC PyTorch base image has the right CUDA 13 / sm_110 build.

## Limitations (being honest)

- **Distilled model only does 8 inference steps** — motion is decent but not buttery smooth. Frame interpolation in post helps.
- **Negative prompts don't work** — the distilled model uses CFG=1.0, which mathematically eliminates the negative prompt term. It accepts the flag silently but does nothing.
- **1080p is the ceiling for quality** — you can generate higher res but the model was trained at 1080p. Above that you get spatial tiling seams and coherence loss. Better to generate at 1080p and upscale.
- **~15 min per clip** — this is a 19B model on an edge device. It's not fast. But it's fully local and offline.

## Hardware

NVIDIA Jetson AGX Thor, JetPack 7.0, CUDA 13.0. 128GB unified memory. The pipeline needs at least 128GB — at 64GB you'd need FP8 + pre-computed text embeddings to fit, and it would be very tight.

If anyone else is running video gen models on Jetson hardware, I'd love to compare notes. The unified memory gotchas are real and basically undocumented.
