#!/usr/bin/env python3
"""Post-production: frame interpolation (RIFE) and spatial upscaling (Real-ESRGAN).

Usage:
    # 2x frame interpolation only (24fps → 48fps)
    python postprocess.py --input video.mp4 --output smooth.mp4 --interpolate 2x

    # 4x interpolation (24fps → 96fps, re-encoded at 60fps)
    python postprocess.py --input video.mp4 --output smooth.mp4 --interpolate 4x --target-fps 60

    # 2x spatial upscale only (1080p → 4K)
    python postprocess.py --input video.mp4 --output upscaled.mp4 --upscale 2x

    # Both: interpolate then upscale
    python postprocess.py --input video.mp4 --output final.mp4 --interpolate 2x --upscale 2x
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def get_video_fps(path):
    """Get video FPS using ffprobe."""
    import json
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", path],
        capture_output=True, text=True
    )
    for s in json.loads(probe.stdout).get("streams", []):
        if s.get("codec_type") == "video":
            r = s.get("r_frame_rate", "24/1")
            num, den = r.split("/")
            return int(num) / int(den)
    return 24.0


def extract_audio(input_path):
    """Extract audio track, return (path, has_audio)."""
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False).name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
             "-vn", "-acodec", "copy", audio_tmp],
            capture_output=True, text=True
        )
        has_audio = result.returncode == 0 and os.path.getsize(audio_tmp) > 0
    except Exception:
        has_audio = False
    return audio_tmp, has_audio


def run_rife(input_path, output_path, exp, target_fps=None, rife_dir="/opt/rife"):
    """Run RIFE frame interpolation using the model directly."""
    import cv2
    import torch
    import torch.nn.functional as F

    multi = 2 ** exp

    print(f"\n{'='*60}")
    print(f"RIFE Frame Interpolation ({multi}x)")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    input_fps = get_video_fps(input_path)
    interp_fps = input_fps * multi
    final_fps = target_fps or interp_fps
    print(f"  Input FPS: {input_fps}")
    print(f"  Interpolated FPS: {interp_fps}")
    if target_fps:
        print(f"  Target FPS: {final_fps}")

    audio_tmp, has_audio = extract_audio(input_path)

    # Load RIFE model
    print("  Loading RIFE model...")
    sys.path.insert(0, rife_dir)
    from train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model(os.path.join(rife_dir, "train_log"), -1)
    model.eval()
    print("  Loaded RIFE v4.25")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read all frames
    cap = cv2.VideoCapture(input_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input: {w}x{h}, {frame_count} frames")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  Read {len(frames)} frames")

    # Pad dimensions to multiple of 32 for RIFE
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    need_pad = (ph != h or pw != w)

    def frame_to_tensor(img):
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        t = t.unsqueeze(0).to(device)
        if need_pad:
            t = F.pad(t, (0, pw - w, 0, ph - h))
        return t

    def tensor_to_frame(t):
        if need_pad:
            t = t[:, :, :h, :w]
        return (t[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

    # Interpolate
    with tempfile.TemporaryDirectory() as tmpdir:
        interp_path = os.path.join(tmpdir, "interp.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(interp_path, fourcc, interp_fps, (w, h))

        with torch.no_grad():
            for i in range(len(frames) - 1):
                # Write original frame
                writer.write(frames[i])

                # Generate interpolated frames
                img0 = frame_to_tensor(frames[i])
                img1 = frame_to_tensor(frames[i + 1])

                if multi == 2:
                    mid = model.inference(img0, img1, timestep=0.5)
                    writer.write(tensor_to_frame(mid))
                elif multi == 4:
                    mid = model.inference(img0, img1, timestep=0.5)
                    q1 = model.inference(img0, mid, timestep=0.5)
                    q3 = model.inference(mid, img1, timestep=0.5)
                    writer.write(tensor_to_frame(q1))
                    writer.write(tensor_to_frame(mid))
                    writer.write(tensor_to_frame(q3))

                if (i + 1) % 20 == 0 or i == 0:
                    print(f"    {i+1}/{len(frames)-1} pairs interpolated", flush=True)

            # Write last frame
            writer.write(frames[-1])
        writer.release()

        # Clean up model
        del model, img0, img1
        if 'mid' in dir():
            del mid
        torch.cuda.empty_cache()

        # Re-encode with ffmpeg for proper codec + optional audio
        ffmpeg_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", interp_path]
        if has_audio:
            ffmpeg_cmd += ["-i", audio_tmp]
        if target_fps and target_fps != interp_fps:
            ffmpeg_cmd += ["-r", str(final_fps)]
        ffmpeg_cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
        if has_audio:
            ffmpeg_cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        ffmpeg_cmd.append(output_path)
        subprocess.run(ffmpeg_cmd, check=True)

    if audio_tmp and os.path.exists(audio_tmp):
        os.remove(audio_tmp)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n  Done: {output_path} ({size_mb:.1f} MB)")
    return True


def run_realesrgan(input_path, output_path, scale, model_dir="/opt/realesrgan"):
    """Run Real-ESRGAN spatial upscaling."""
    print(f"\n{'='*60}")
    print(f"Real-ESRGAN Spatial Upscale ({scale}x)")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Extract audio
    audio_tmp = None
    has_audio = False
    try:
        audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False).name
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
             "-vn", "-acodec", "copy", audio_tmp],
            capture_output=True, text=True
        )
        has_audio = result.returncode == 0 and os.path.getsize(audio_tmp) > 0
    except Exception:
        has_audio = False

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract frames
        frames_dir = os.path.join(tmpdir, "frames")
        upscaled_dir = os.path.join(tmpdir, "upscaled")
        os.makedirs(frames_dir)
        os.makedirs(upscaled_dir)

        # Get fps
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", input_path],
            capture_output=True, text=True
        )
        import json
        fps = 24
        for s in json.loads(probe.stdout).get("streams", []):
            if s.get("codec_type") == "video":
                r = s.get("r_frame_rate", "24/1")
                num, den = r.split("/")
                fps = int(num) / int(den)
                break

        print(f"  Extracting frames...")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
            os.path.join(frames_dir, "frame_%06d.png")
        ], check=True)

        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        print(f"  Extracted {frame_count} frames")

        # Run Real-ESRGAN
        print(f"  Upscaling {frame_count} frames at {scale}x...")
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
        import cv2
        import glob

        # RealESRGAN_x2plus for 2x, RealESRGAN_x4plus for 4x
        if scale <= 2:
            model_name = "RealESRGAN_x2plus"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            model_name = "RealESRGAN_x4plus"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            netscale = 4

        # Download model weights if needed
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        if not os.path.exists(model_path):
            os.makedirs(model_dir, exist_ok=True)
            url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{model_name}.pth"
            print(f"  Downloading model: {model_name}...")
            import urllib.request
            urllib.request.urlretrieve(url, model_path)

        upsampler = RealESRGANer(
            scale=netscale, model_path=model_path, model=model,
            tile=512, tile_pad=10, pre_pad=0, half=True,
            device="cuda"
        )

        frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        for i, frame_path in enumerate(frames):
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=scale)
            out_path = os.path.join(upscaled_dir, os.path.basename(frame_path))
            cv2.imwrite(out_path, output)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    {i+1}/{len(frames)} frames", flush=True)

        print(f"  Encoding video...")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(upscaled_dir, "frame_%06d.png"),
        ]
        if has_audio:
            ffmpeg_cmd += ["-i", audio_tmp]
        ffmpeg_cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        ]
        if has_audio:
            ffmpeg_cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        ffmpeg_cmd.append(output_path)
        subprocess.run(ffmpeg_cmd, check=True)

        # Cleanup
        del upsampler
        torch.cuda.empty_cache()

    if audio_tmp and os.path.exists(audio_tmp):
        os.remove(audio_tmp)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n  Done: {output_path} ({size_mb:.1f} MB)")
    return True


def main():
    p = argparse.ArgumentParser(description="Post-production: interpolation + upscaling")
    p.add_argument("--input", "-i", required=True, help="Input video path")
    p.add_argument("--output", "-o", required=True, help="Output video path")
    p.add_argument("--interpolate", choices=["2x", "4x"], default=None,
                   help="Frame interpolation factor (2x or 4x)")
    p.add_argument("--upscale", choices=["2x", "4x"], default=None,
                   help="Spatial upscale factor (2x or 4x)")
    p.add_argument("--target-fps", type=int, default=None,
                   help="Re-encode at this FPS after interpolation (e.g. 60)")
    p.add_argument("--rife-dir", type=str, default="/opt/rife",
                   help="Path to RIFE model directory (default: /opt/rife)")
    p.add_argument("--realesrgan-dir", type=str, default="/opt/realesrgan",
                   help="Path to Real-ESRGAN model directory (default: /opt/realesrgan)")
    args = p.parse_args()

    if not args.interpolate and not args.upscale:
        print("ERROR: Specify at least --interpolate or --upscale")
        sys.exit(1)

    current_input = args.input

    # Step 1: Frame interpolation (if requested)
    if args.interpolate:
        exp = 1 if args.interpolate == "2x" else 2
        if args.upscale:
            # Intermediate output
            interp_out = args.output.replace(".mp4", "_interp.mp4")
        else:
            interp_out = args.output

        if not run_rife(current_input, interp_out, exp, args.target_fps, rife_dir=args.rife_dir):
            sys.exit(1)
        current_input = interp_out

    # Step 2: Spatial upscaling (if requested)
    if args.upscale:
        scale = 2 if args.upscale == "2x" else 4
        if not run_realesrgan(current_input, args.output, scale, model_dir=args.realesrgan_dir):
            sys.exit(1)

        # Clean up intermediate if we did both steps
        if args.interpolate and current_input != args.input:
            os.remove(current_input)

    print(f"\nAll done: {args.output}")


if __name__ == "__main__":
    main()
