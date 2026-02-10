#!/usr/bin/env python3
"""LTX-2 video generation with FP8 quantization and LoRA support.

Load order is critical: base model → LoRA → FP8 casting.
Reversing this breaks LoRA (see diffusers issue #11648).

Distilled model uses 8 steps, guidance_scale=1.0, and sigma values from diffusers.
Prompts are plain natural language paragraphs — no structured tags.

Chunked VAE decode prevents OOM on large generations (e.g. 1920x1088 @ 361 frames).
"""

import argparse
import gc
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="LTX-2 Video Generation")
    p.add_argument("--prompt", type=str, required=True,
                    help="Plain natural language prompt paragraph")
    p.add_argument("--negative-prompt", type=str, default="",
                    help="Negative prompt")
    p.add_argument("--model", type=str,
                    default="rootonchair/LTX-2-19b-distilled",
                    help="HF model ID (diffusers format)")
    p.add_argument("--output", type=str, default="/outputs/video.mp4",
                    help="Output video path")

    # Generation params — distilled defaults
    p.add_argument("--steps", type=int, default=8,
                    help="Inference steps (8 for distilled)")
    p.add_argument("--guidance-scale", type=float, default=1.0,
                    help="Guidance scale (1.0 for distilled)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-frames", type=int, default=97,
                    help="Number of frames (97 ~ 4s at 24fps)")
    p.add_argument("--seed", type=int, default=None)

    # FP8 quantization (on by default)
    p.add_argument("--no-fp8", action="store_true",
                    help="Disable FP8 quantization")

    # Image-to-video
    p.add_argument("--image", type=str, default=None,
                    help="Path or URL to input image for image-to-video mode")

    # LoRA
    p.add_argument("--lora", type=str, default=None,
                    help="HF LoRA repo ID")
    p.add_argument("--lora-weight-name", type=str, default=None,
                    help="Specific weight filename in LoRA repo")
    p.add_argument("--lora-scale", type=float, default=1.0,
                    help="LoRA adapter scale (0.0-1.0)")

    # VAE chunking
    p.add_argument("--vae-chunk-frames", type=int, default=None,
                    help="Decode VAE in chunks of N latent frames "
                         "(auto-calculated if not set; reduces peak VRAM)")

    # Memory optimization
    p.add_argument("--sequential-offload", action="store_true",
                    help="Use sequential CPU offload (per-layer instead of per-model). "
                         "Slower but uses much less peak GPU memory. "
                         "Use for high-res/long generations that OOM.")

    return p.parse_args()



def flush_memory(label="", drop_page_cache=False):
    """Free reclaimable memory on Jetson Thor.

    On unified memory (CPU/GPU share RAM), moving tensors to CPU does nothing.
    We must actually delete objects, flush CUDA cache, and run GC.

    IMPORTANT: drop_page_cache should only be True AFTER all models are done
    and the video is written to disk. On unified memory, model weights loaded
    via safetensors mmap may still be backed by page cache pages even after
    .to("cuda"). Dropping the page cache while models are still needed will
    evict their weights and cause segfaults.
    """
    torch.cuda.empty_cache()
    gc.collect()
    # Second pass — GC can reveal more garbage after first pass
    torch.cuda.empty_cache()
    gc.collect()

    # Flush kernel page cache (mmap'd weight files linger here)
    # Only safe when ALL models are deleted and video is on disk.
    if drop_page_cache:
        try:
            subprocess.run(["sync"], check=False)
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        except (PermissionError, OSError):
            pass

    # Report
    cuda_alloc = torch.cuda.memory_allocated() / 1e9
    cuda_reserved = torch.cuda.memory_reserved() / 1e9
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if parts[0].rstrip(':') in ("MemTotal", "MemFree", "MemAvailable", "Cached", "Buffers"):
                    meminfo[parts[0].rstrip(':')] = int(parts[1]) // 1024  # KB -> MB
        sys_avail = meminfo.get("MemAvailable", 0)
        sys_cached = meminfo.get("Cached", 0)
        print(f"  [{label}] CUDA alloc={cuda_alloc:.1f}GB reserved={cuda_reserved:.1f}GB | "
              f"SYS avail={sys_avail}MB cached={sys_cached}MB", flush=True)
    except Exception:
        print(f"  [{label}] CUDA alloc={cuda_alloc:.1f}GB reserved={cuda_reserved:.1f}GB", flush=True)


def estimate_vae_chunk_size(height, width, num_frames):
    """Estimate safe VAE chunk size based on resolution and frame count.

    The VAE temporal compression ratio for LTX-2 is 8:1 (8 pixel frames → 1 latent frame).
    Latent spatial dims are height/32 × width/32.

    At 1920x1088 the VAE decode of the full 361-frame latent tensor (~46 latent frames)
    requires ~12+ GB just for intermediates. Chunking to ~8 latent frames keeps peak
    usage manageable on the Thor's 128GB unified memory.
    """
    pixels_per_frame = height * width
    latent_frames = (num_frames - 1) // 8 + 1  # temporal compression ratio

    # Single-pass decode — no manual chunking. The VAE's built-in tiling
    # (spatial 512x512, temporal 16-frame tiles) handles memory internally.
    #
    # This requires ~60-80 GB for 1080p @ 161 frames, which fits on Jetson
    # Thor (128 GB unified) now that we properly free the transformer and
    # text encoder before decode. Manual chunking
    # introduces visible stitch artifacts (doubling/stutter at chunk
    # boundaries from linear crossfade blending).
    #
    # If you're running on a system with less memory (e.g. 64 GB VRAM),
    # override on the command line:
    #   --vae-chunk-frames 8    (3 chunks @ 161 frames, 2 stitch points)
    #   --vae-chunk-frames 4    (6 chunks @ 161 frames, 5 stitch points)
    # Fewer chunks = fewer artifacts but more memory.
    return latent_frames


def _flush_frames_to_raw(raw_file, tensor):
    """Convert bfloat16 video tensor [1, C, T, H, W] to uint8 RGB and write to raw file.

    Returns (num_frames_written, height, width).
    """
    video = tensor.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
    video = ((video.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).numpy()
    raw_file.write(video.tobytes())
    t, h, w, _ = video.shape
    del video
    return t, h, w


def chunked_vae_decode(pipe, latents, chunk_size, overlap=2):
    """Decode latents through VAE in temporal chunks, streaming frames to disk.

    Uses overlapping chunks with linear crossfade blending at boundaries.
    Finalized frames are converted to uint8 and written to a raw RGB file
    immediately, so only the blend tail (~16 frames) stays in memory.

    Args:
        pipe: The LTX2Pipeline with VAE
        latents: Full latent tensor [B, C, T, H, W]
        chunk_size: Number of latent temporal frames per chunk
        overlap: Extra latent frames on each side for temporal context

    Returns:
        (raw_file_path, frame_count, height, width) tuple.
        raw_file_path points to a raw RGB24 file (H*W*3 bytes per frame).
    """
    vae = pipe.vae
    total_latent_frames = latents.shape[2]

    # Cast latents to match VAE weight dtype (pipeline returns float32, VAE is bfloat16)
    vae_dtype = next(vae.parameters()).dtype
    if latents.dtype != vae_dtype:
        print(f"  Casting latents from {latents.dtype} to {vae_dtype}")
        latents = latents.to(dtype=vae_dtype)

    raw_fd = tempfile.NamedTemporaryFile(suffix='.raw', delete=False,
                                         dir=os.path.dirname("/outputs/"))
    raw_path = raw_fd.name
    frame_count = 0
    out_h = out_w = 0

    if total_latent_frames <= chunk_size:
        print(f"  VAE decode: single pass ({total_latent_frames} latent frames)")
        flush_memory("pre-vae-decode")
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]
        flush_memory("post-vae-forward")
        n, out_h, out_w = _flush_frames_to_raw(raw_fd, decoded.cpu())
        frame_count = n
        del decoded
        raw_fd.close()
        return raw_path, frame_count, out_h, out_w

    print(f"  VAE decode: chunked ({total_latent_frames} latent frames, "
          f"chunk_size={chunk_size}, overlap={overlap})")

    blend_tail = None   # bfloat16 [1, C, T_tail, H, W] — only part kept in RAM

    chunk_ranges = [(s, min(s + chunk_size, total_latent_frames))
                    for s in range(0, total_latent_frames, chunk_size)]

    for idx, (start, end) in enumerate(chunk_ranges):
        is_last = (idx == len(chunk_ranges) - 1)

        # Extend chunk with overlap for temporal context at boundaries
        ctx_start = max(0, start - overlap)
        ctx_end = min(total_latent_frames, end + overlap)

        chunk = latents[:, :, ctx_start:ctx_end, :, :]
        print(f"    Decoding latent frames {ctx_start}-{ctx_end-1} "
              f"(core {start}-{end-1}, {end - start}/{total_latent_frames})...")

        with torch.no_grad():
            decoded_chunk = vae.decode(chunk, return_dict=False)[0]

        # Figure out pixel boundaries for the overlap regions
        total_ctx_latent = ctx_end - ctx_start
        total_decoded_pixels = decoded_chunk.shape[2]
        pixels_per_latent = total_decoded_pixels / total_ctx_latent

        left_ctx_latent = start - ctx_start
        right_ctx_latent = ctx_end - end
        left_ctx_pixels = round(left_ctx_latent * pixels_per_latent)
        right_ctx_pixels = round(right_ctx_latent * pixels_per_latent)

        # Strip right context
        if right_ctx_pixels > 0:
            decoded_chunk = decoded_chunk[:, :, :(-right_ctx_pixels), :, :]

        decoded_chunk = decoded_chunk.cpu()
        del chunk

        # --- Blend with previous tail ---
        if blend_tail is None:
            current = decoded_chunk
        else:
            blend_pixels = left_ctx_pixels
            if blend_pixels > 0 and blend_tail.shape[2] >= blend_pixels:
                weights = torch.linspace(0.0, 1.0, blend_pixels,
                                         dtype=decoded_chunk.dtype)
                weights = weights.reshape(1, 1, -1, 1, 1)

                blended = ((1.0 - weights) * blend_tail[:, :, -blend_pixels:, :, :]
                           + weights * decoded_chunk[:, :, :blend_pixels, :, :])

                current = torch.cat([
                    blend_tail[:, :, :-blend_pixels, :, :],
                    blended,
                    decoded_chunk[:, :, blend_pixels:, :, :],
                ], dim=2)
            else:
                current = torch.cat([blend_tail, decoded_chunk], dim=2)
            del blend_tail

        del decoded_chunk

        # --- Flush finalized frames to disk, keep tail for next blend ---
        if not is_last:
            tail_keep = round(overlap * pixels_per_latent)
            tail_keep = max(tail_keep, 1)

            if current.shape[2] > tail_keep:
                finalized = current[:, :, :-tail_keep, :, :]
                blend_tail = current[:, :, -tail_keep:, :, :].clone()
                n, out_h, out_w = _flush_frames_to_raw(raw_fd, finalized)
                frame_count += n
                del finalized
            else:
                blend_tail = current
        else:
            # Last chunk — flush everything
            n, out_h, out_w = _flush_frames_to_raw(raw_fd, current)
            frame_count += n
            del current
            blend_tail = None

        flush_memory(f"vae chunk {start}-{end-1}")

    raw_fd.close()
    print(f"  Streamed {frame_count} frames to disk ({os.path.getsize(raw_path) / 1e6:.0f} MB)")
    return raw_path, frame_count, out_h, out_w


def decode_audio(pipe, audio_latents):
    """Decode audio latents through audio VAE + vocoder to waveform.

    When output_type="latent", the pipeline returns audio latents that are
    already denormalized and unpacked but NOT decoded. We run them through
    the audio VAE to get mel spectrograms, then through the vocoder to get
    a 24kHz stereo waveform.

    Returns:
        (waveform, sample_rate) tuple or (None, None) if audio components missing.
        Waveform is [channels, samples] on CPU.
    """
    if not hasattr(pipe, 'audio_vae') or pipe.audio_vae is None:
        print("  No audio VAE found — skipping audio")
        return None, None
    if not hasattr(pipe, 'vocoder') or pipe.vocoder is None:
        print("  No vocoder found — skipping audio")
        return None, None

    # Grab sample rate NOW before we delete anything
    sample_rate = pipe.vocoder.config.output_sampling_rate

    print("  Decoding audio latents...", flush=True)
    print(f"    Audio latents shape: {audio_latents.shape}, dtype: {audio_latents.dtype}", flush=True)

    with torch.no_grad():
        audio_latents_gpu = audio_latents.to(dtype=pipe.audio_vae.dtype, device="cuda")
        # Free the original copy immediately
        del audio_latents
        flush_memory("audio: freed input latents")

        mel = pipe.audio_vae.decode(audio_latents_gpu, return_dict=False)[0]
        print(f"    Mel spectrogram shape: {mel.shape}, dtype: {mel.dtype}", flush=True)

        # Free audio_latents from GPU before vocoder runs
        del audio_latents_gpu
        flush_memory("audio: freed gpu latents, pre-vocoder")

        waveform = pipe.vocoder(mel)
        print(f"    Waveform shape: {waveform.shape}, dtype: {waveform.dtype}", flush=True)

        # Free mel immediately
        del mel
        flush_memory("audio: freed mel, post-vocoder")

    # Move to CPU, squeeze batch dim, free GPU copy
    waveform_cpu = waveform.float().cpu()
    del waveform
    flush_memory("audio: waveform moved to CPU")
    waveform = waveform_cpu

    if waveform.ndim == 3:
        waveform = waveform[0]  # [batch, channels, samples] -> [channels, samples]

    return waveform, sample_rate


def save_audio_wav(waveform, path, sample_rate=24000):
    """Save a waveform tensor as a WAV file using numpy (no scipy needed)."""
    import struct
    audio_np = waveform.numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]  # mono -> [1, samples]
    channels, num_samples = audio_np.shape
    # Clamp and convert to int16
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    # Interleave channels
    interleaved = audio_int16.T.flatten()
    raw_bytes = interleaved.tobytes()

    with open(path, 'wb') as f:
        # WAV header
        data_size = len(raw_bytes)
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # PCM
        f.write(struct.pack('<H', channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * channels * 2))  # byte rate
        f.write(struct.pack('<H', channels * 2))  # block align
        f.write(struct.pack('<H', 16))  # bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(raw_bytes)


def mux_audio_video(video_path, audio_path, output_path):
    """Combine video and audio into a single MP4 using ffmpeg."""
    tmp_output = output_path + ".muxing.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        tmp_output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg mux failed: {result.stderr}")
        return False
    os.replace(tmp_output, output_path)
    return True


def save_latents_backup(latents, output_path):
    """Save raw latents as numpy — can be decoded later if video export fails."""
    backup_path = output_path.replace('.mp4', '_latents.npz')
    try:
        lat_np = latents.cpu().float().numpy()
        np.savez_compressed(backup_path, latents=lat_np)
        size_mb = os.path.getsize(backup_path) / 1e6
        print(f"  Latents backed up: {backup_path} (shape={lat_np.shape}, {size_mb:.1f} MB)")
        del lat_np
        gc.collect()
        return backup_path
    except Exception as e:
        print(f"  WARNING: Latent backup failed: {e}")
        return None


def save_frames_backup(frames, output_path):
    """Save decoded frames as numpy backup."""
    backup_path = output_path.replace('.mp4', '_frames.npz')
    try:
        frame_arrays = np.stack([np.array(f) for f in frames])
        np.savez_compressed(backup_path, frames=frame_arrays)
        size_mb = os.path.getsize(backup_path) / 1e6
        print(f"  Frames backed up: {backup_path} ({frame_arrays.shape}, {size_mb:.1f} MB)")
        del frame_arrays
        gc.collect()
        return backup_path
    except Exception as e:
        print(f"  WARNING: Frame backup failed: {e}")
        return None


def main():
    args = parse_args()
    use_fp8 = not args.no_fp8
    device = "cuda"
    latent_backup_path = None
    frame_backup_path = None

    # Add datetime stamp to output filename
    base, ext = os.path.splitext(args.output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = f"{base}_{stamp}{ext}"

    # Ensure output directory exists early
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"=" * 60)
    print(f"LTX-2 Video Generation")
    print(f"  Model:      {args.model}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames:     {args.num_frames} ({args.num_frames / 24:.1f}s at 24fps)")
    print(f"  Steps:      {args.steps}")
    print(f"  FP8:        {use_fp8}")
    print(f"  LoRA:       {args.lora or 'none'}")
    print(f"  Seed:       {args.seed or 'random'}")
    print(f"  Output:     {args.output}")
    print(f"=" * 60)

    # --- Step 1: Load base model ---
    is_i2v = args.image is not None
    if is_i2v:
        from diffusers import LTX2ImageToVideoPipeline as PipeClass
        from diffusers.utils import load_image
        print(f"  Mode: image-to-video (input: {args.image})")
    else:
        from diffusers import LTX2Pipeline as PipeClass
        print(f"  Mode: text-to-video")

    pipe = PipeClass.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    # --- Step 2: Load LoRA (BEFORE FP8 casting) ---
    if args.lora:
        print(f"Loading LoRA: {args.lora} (scale={args.lora_scale})")
        lora_kwargs = {"adapter_name": "lora"}
        if args.lora_weight_name:
            lora_kwargs["weight_name"] = args.lora_weight_name
        pipe.load_lora_weights(args.lora, **lora_kwargs)
        pipe.set_adapters(["lora"], [args.lora_scale])
        print("LoRA loaded successfully")

    # --- Step 3: FP8 quantization (AFTER LoRA) ---
    if use_fp8:
        print("Casting transformer to float8_e4m3fn...")
        for name, param in pipe.transformer.named_parameters():
            if param.ndim >= 2:  # Only quantize weight matrices
                param.data = param.data.to(torch.float8_e4m3fn)
        print("FP8 quantization applied")

    if args.sequential_offload:
        print("Using sequential (per-layer) CPU offload — lower peak memory, slower")
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        # On Jetson Thor (unified memory), CPU offload is pointless and harmful:
        # CPU/GPU share the same RAM, so .to("cpu") frees nothing. The hooks
        # just create reference paths that prevent model deletion, and removing
        # them later leaves models in an inconsistent device state.
        # Instead, load everything onto CUDA directly. 128GB fits all models.
        pipe.to(device)

    # Fix: Monkey-patch scheduler to prevent off-by-one IndexError on last step.
    # FlowMatchEulerDiscreteScheduler._init_step_index() uses float equality to
    # find the starting step index. With use_dynamic_shifting=True (full LTX-2),
    # shifted sigmas can create near-duplicate timestep values, causing it to
    # return index 1 instead of 0. This shifts ALL step indices up by 1, so the
    # last step reads sigmas[N+1] which is out of bounds.
    # See: https://github.com/huggingface/diffusers/issues/9331
    import types

    def _safe_init_step_index(self, timestep):
        self._step_index = self._begin_index if self._begin_index is not None else 0

    pipe.scheduler._init_step_index = types.MethodType(_safe_init_step_index, pipe.scheduler)

    # --- Build generation kwargs ---
    gen_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        output_type="latent",  # <-- KEY: get raw latents, decode ourselves
        frame_rate=24.0,      # Used to compute audio temporal length
    )

    # Image-to-video: load and pass input image
    if is_i2v:
        gen_kwargs["image"] = load_image(args.image)
        print(f"  Input image loaded: {args.image}")

    # Distilled sigma schedule — only for distilled models.
    # The full model uses FlowMatchEulerDiscreteScheduler's default sigmas.
    is_distilled = "distilled" in args.model.lower()
    if is_distilled:
        try:
            from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
            gen_kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
            print(f"Using distilled sigma schedule ({len(DISTILLED_SIGMA_VALUES)} values)")
        except ImportError:
            print("WARNING: DISTILLED_SIGMA_VALUES not found, using default schedule")
    else:
        # Full model: let the pipeline calculate sigmas via its default
        # np.linspace(1.0, 1/steps, steps) schedule with dynamic shifting.
        # The _init_step_index monkey-patch above prevents the off-by-one crash.
        print(f"Using full model default sigma schedule ({args.steps} steps, "
              f"guidance_scale={args.guidance_scale})")

    if args.seed is not None:
        gen_kwargs["generator"] = torch.Generator(device=device).manual_seed(args.seed)

    # --- Generate latents ---
    print(f"\n[1/5] Running diffusion ({args.steps} steps)...")
    pipe.vae.enable_tiling()

    output = pipe(**gen_kwargs)
    print("  DEBUG: pipe() returned, accessing output.frames...", flush=True)
    video_latents = output.frames
    print(f"  DEBUG: got video_latents, shape={video_latents.shape}", flush=True)
    audio_latents = output.audio
    print(f"  DEBUG: got audio_latents, is_none={audio_latents is None}", flush=True)

    # Free the output container and generation kwargs — they hold references
    del output, gen_kwargs
    gc.collect()

    # Free the transformer and text encoder — no longer needed after diffusion.
    # No hooks to worry about since we loaded directly onto CUDA with pipe.to().
    #
    # CRITICAL: If LoRA was loaded, peft wraps transformer layers with adapter
    # layers that create reference cycles. del pipe.transformer alone won't free
    # the ~38GB of weights — peft's wrappers keep them alive. Fusing + unloading
    # removes the peft wrappers so GC can actually reclaim the memory.
    if args.lora:
        print("  Fusing LoRA weights before cleanup...", flush=True)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    print("  Freeing transformer...", flush=True)
    del pipe.transformer
    pipe.transformer = None
    print("  Freeing text_encoder...", flush=True)
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None
    if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
        del pipe.text_encoder_2
        pipe.text_encoder_2 = None
    if hasattr(pipe, 'tokenizer') and pipe.tokenizer is not None:
        del pipe.tokenizer
        pipe.tokenizer = None
    if hasattr(pipe, 'tokenizer_2') and pipe.tokenizer_2 is not None:
        del pipe.tokenizer_2
        pipe.tokenizer_2 = None
    if hasattr(pipe, 'scheduler') and pipe.scheduler is not None:
        del pipe.scheduler
        pipe.scheduler = None
    # Connectors bridge audio/video during diffusion — not needed after.
    # They're in _lora_loadable_modules so peft may have touched them.
    if hasattr(pipe, 'connectors') and pipe.connectors is not None:
        print("  Freeing connectors...", flush=True)
        del pipe.connectors
        pipe.connectors = None
    flush_memory("post-diffusion")

    print(f"  Video latent shape: {video_latents.shape}", flush=True)
    print(f"  Video latent dtype: {video_latents.dtype}", flush=True)
    if audio_latents is not None:
        print(f"  Audio latent shape: {audio_latents.shape}", flush=True)
    else:
        print("  Audio latents: None", flush=True)

    latents = video_latents
    del video_latents  # single reference from here on

    # --- Save latent backup IMMEDIATELY (tiny compared to frames) ---
    print(f"\n[2/5] Backing up latents...", flush=True)
    latent_backup_path = save_latents_backup(latents, args.output)
    print(f"  DEBUG: latent backup done", flush=True)

    # --- Decode audio (lightweight, do before heavy video decode) ---
    print(f"\n[3/5] Audio decode...", flush=True)
    audio_waveform = None
    audio_sample_rate = None
    try:
        audio_waveform, audio_sample_rate = decode_audio(pipe, audio_latents)
        if audio_waveform is not None:
            duration = audio_waveform.shape[-1] / audio_sample_rate
            print(f"  Audio: {duration:.1f}s, {audio_waveform.shape[0]}ch, {audio_sample_rate}Hz")
    except Exception as e:
        print(f"  WARNING: Audio decode failed: {e}")
        traceback.print_exc()
    # audio_latents was already freed inside decode_audio, but ensure all audio components gone
    if hasattr(pipe, 'audio_vae') and pipe.audio_vae is not None:
        del pipe.audio_vae
        pipe.audio_vae = None
    if hasattr(pipe, 'vocoder') and pipe.vocoder is not None:
        del pipe.vocoder
        pipe.vocoder = None
    if hasattr(pipe, 'audio_projection') and pipe.audio_projection is not None:
        del pipe.audio_projection
        pipe.audio_projection = None
    flush_memory("post-audio")

    # --- Chunked VAE decode (streams frames to disk) ---
    print(f"\n[4/5] VAE decode...", flush=True)
    chunk_size = args.vae_chunk_frames or estimate_vae_chunk_size(
        args.height, args.width, args.num_frames
    )

    raw_path = None
    audio_tmp = None
    try:
        raw_path, frame_count, dec_h, dec_w = chunked_vae_decode(
            pipe, latents, chunk_size)

        # Free everything — frames are on disk now
        del latents
        del pipe
        flush_memory("post-vae-decode", drop_page_cache=True)

        print(f"  Decoded {frame_count} frames ({dec_w}x{dec_h})", flush=True)

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n  ERROR: GPU OOM during VAE decode!")
        print(f"  {e}")
        print(f"  Latent backup saved at: {latent_backup_path}")
        print(f"  You can decode later with a smaller --vae-chunk-frames value.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR during VAE decode: {e}")
        traceback.print_exc()
        print(f"  Latent backup saved at: {latent_backup_path}")
        sys.exit(1)

    # --- Encode video directly from raw frames via ffmpeg ---
    print(f"\n[5/5] Encoding video...")

    try:
        # Build ffmpeg command — encode raw RGB frames to H.264
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{dec_w}x{dec_h}", "-r", "24",
            "-i", raw_path,
        ]

        # If we have audio, mux it in the same ffmpeg pass
        audio_tmp = None
        if audio_waveform is not None and audio_sample_rate is not None:
            audio_tmp = args.output.replace('.mp4', '_audio.wav')
            save_audio_wav(audio_waveform, audio_tmp, audio_sample_rate)
            del audio_waveform
            gc.collect()
            ffmpeg_cmd += ["-i", audio_tmp]

        ffmpeg_cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        ]

        if audio_tmp:
            ffmpeg_cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

        video_tmp = args.output + ".encoding.mp4"
        ffmpeg_cmd.append(video_tmp)

        print(f"  Encoding {frame_count} frames with ffmpeg...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        os.replace(video_tmp, args.output)

        if audio_tmp:
            print(f"  Audio muxed successfully ({audio_sample_rate}Hz stereo)")

        size_mb = os.path.getsize(args.output) / 1e6
        print(f"\n  ✓ Video saved: {args.output} ({size_mb:.1f} MB)")

        # Clean up backups on success
        if latent_backup_path and os.path.exists(latent_backup_path):
            os.remove(latent_backup_path)
            print(f"  Cleaned up: {latent_backup_path}")

    except Exception as e:
        print(f"\n  ERROR during video encoding: {e}")
        traceback.print_exc()
        if latent_backup_path and os.path.exists(latent_backup_path):
            print(f"  Latent backup preserved: {latent_backup_path}")
        if raw_path and os.path.exists(raw_path):
            print(f"  Raw frames preserved: {raw_path}")
            print(f"  To encode manually:")
            print(f"    ffmpeg -f rawvideo -pix_fmt rgb24 -s {dec_w}x{dec_h} -r 24 "
                  f"-i {raw_path} -c:v libx264 -pix_fmt yuv420p {args.output}")

    finally:
        # Clean up temp files
        if audio_tmp and os.path.exists(audio_tmp):
            os.remove(audio_tmp)
        if raw_path and os.path.exists(raw_path):
            os.remove(raw_path)
            print(f"  Cleaned up raw frames: {raw_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
