#!/usr/bin/env python3
"""Decode saved latent backups from failed/OOM runs into video.

Usage:
    python3 decode_latents.py --input video_latents.npz --output video.mp4
    python3 decode_latents.py --input video_latents.npz --output video.mp4 --vae-chunk-frames 4
"""

import argparse
import gc
import os
import sys

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description="Decode LTX-2 latent backups to video")
    p.add_argument("--input", type=str, required=True, help="Path to _latents.npz file")
    p.add_argument("--output", type=str, required=True, help="Output .mp4 path")
    p.add_argument("--model", type=str, default="rootonchair/LTX-2-19b-distilled",
                    help="HF model ID (needs matching VAE)")
    p.add_argument("--vae-chunk-frames", type=int, default=4,
                    help="Latent frames per decode chunk (lower = less VRAM)")
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    print(f"Loading latents from {args.input}...")
    data = np.load(args.input)
    latents = torch.from_numpy(data["latents"]).to(torch.bfloat16)
    print(f"  Shape: {latents.shape}")

    print(f"Loading VAE from {args.model}...")
    from diffusers import AutoencoderKLLTX2Video
    vae = AutoencoderKLLTX2Video.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.enable_tiling()
    vae.to("cuda")

    total_latent_frames = latents.shape[2]
    chunk_size = args.vae_chunk_frames
    overlap = 2  # extra latent frames on each side for temporal context

    print(f"Decoding {total_latent_frames} latent frames in chunks of {chunk_size} (overlap={overlap})...")

    video = None
    video_frames = 0

    for start in range(0, total_latent_frames, chunk_size):
        end = min(start + chunk_size, total_latent_frames)

        # Extend with overlap for temporal context at boundaries
        ctx_start = max(0, start - overlap)
        ctx_end = min(total_latent_frames, end + overlap)

        chunk = latents[:, :, ctx_start:ctx_end, :, :].to("cuda")
        print(f"  Decoding {ctx_start}-{ctx_end-1} (core {start}-{end-1})...")

        with torch.no_grad():
            decoded = vae.decode(chunk, return_dict=False)[0]

        # Figure out pixel boundaries for the overlap regions
        total_ctx_latent = ctx_end - ctx_start
        total_decoded_pixels = decoded.shape[2]
        pixels_per_latent = total_decoded_pixels / total_ctx_latent

        left_ctx_latent = start - ctx_start
        right_ctx_latent = ctx_end - end
        left_ctx_pixels = round(left_ctx_latent * pixels_per_latent)
        right_ctx_pixels = round(right_ctx_latent * pixels_per_latent)

        # Strip right context
        if right_ctx_pixels > 0:
            decoded = decoded[:, :, :(-right_ctx_pixels), :, :]

        decoded = decoded.cpu()
        del chunk

        if video is None:
            video = decoded
            video_frames = decoded.shape[2]
        else:
            # Linear crossfade blend in the left overlap region
            blend_pixels = left_ctx_pixels
            if blend_pixels > 0 and video_frames >= blend_pixels:
                weights = torch.linspace(0.0, 1.0, blend_pixels,
                                         dtype=decoded.dtype)
                weights = weights.reshape(1, 1, -1, 1, 1)

                prev_tail = video[:, :, -blend_pixels:, :, :]
                curr_head = decoded[:, :, :blend_pixels, :, :]
                blended = (1.0 - weights) * prev_tail + weights * curr_head

                video = torch.cat([
                    video[:, :, :-blend_pixels, :, :],
                    blended,
                    decoded[:, :, blend_pixels:, :, :],
                ], dim=2)
            else:
                video = torch.cat([video, decoded], dim=2)

            video_frames = video.shape[2]

        del decoded
        torch.cuda.empty_cache()
        gc.collect()

    # Post-process to PIL
    video = video.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
    video = ((video.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).numpy()

    from PIL import Image
    from diffusers.utils import export_to_video
    frames = [Image.fromarray(video[i]) for i in range(video.shape[0])]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_to_video(frames, args.output, fps=args.fps)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n✓ Video saved: {args.output} ({size_mb:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
