# Post-Processing Pipeline

Smooth and upscale AI-generated videos using RIFE (frame interpolation) and Real-ESRGAN (spatial upscaling). Everything runs in Docker — your system Python is never touched.

## Setup (One-Time)

### 1. Download RIFE Model Weights

The RIFE v4.25 model weights must be downloaded manually from Google Drive (they're ~23MB):

1. Download from: https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg
2. Extract the zip:
   ```bash
   mkdir -p rife_model
   cd rife_model
   unzip RIFEv4.25_0919.zip
   # This creates rife_model/train_log/ with:
   #   flownet.pkl    — pretrained weights (24MB)
   #   RIFE_HDv3.py   — model architecture
   #   IFNet_HDv3.py  — flow estimation network
   #   refine.py      — refinement module
   ```

### 2. Build Docker Image

```bash
docker build -f Dockerfile.postprocess -t ltx2-postprocess .
```

**NGC container compatibility notes** (already handled in the Dockerfile):
- `libgl1-mesa-glx` was renamed to `libgl1-mesa-dev` in Ubuntu 24.04 (Noble)
- NGC's pip ships a ghost `/etc/pip/constraint.txt` reference that must be cleared
- `basicsr` imports the removed `torchvision.transforms.functional_tensor` — patched via `sed`
- RIFE's `inference_video.py` depends on `skvideo` which uses the removed `np.float` — we bypass this by calling the RIFE model directly with OpenCV instead

## Usage

### Single Video

```bash
# Smooth out motion (2x frame interpolation, 24fps → 48fps)
./postprocess.sh outputs/video.mp4 outputs/video_smooth.mp4 --interpolate 2x

# Smooth + convert to 60fps
./postprocess.sh outputs/video.mp4 outputs/video_60fps.mp4 --interpolate 4x --target-fps 60

# Upscale 1080p → 4K
./postprocess.sh outputs/video.mp4 outputs/video_4k.mp4 --upscale 2x

# Both: interpolate first, then upscale
./postprocess.sh outputs/video.mp4 outputs/video_final.mp4 --interpolate 2x --upscale 2x
```

### Batch Processing

```bash
# Process all batch1 videos (interpolate 2x + upscale 2x)
./postprocess_batch1.sh
# Input:  outputs/batch1/*.mp4
# Output: outputs/batch1_postprocessed/*.mp4
```

The batch script processes videos sequentially: for each video, it interpolates then upscales before moving to the next. This avoids storing all intermediate files at once. Already-completed videos are skipped on re-run.

## What Each Step Does

### Frame Interpolation (RIFE v4.25)
- **Purpose**: Fills in motion between frames to reduce the "clippy" AI-generated look
- **How**: Neural optical flow estimation — RIFE predicts the motion field between consecutive frames, then synthesizes intermediate frames at the midpoints
- **Model**: Practical-RIFE v4.25 (recommended by authors for post-processing generated video)
- **2x**: 160 frames → 319 frames, 24fps → 48fps
- **4x**: 160 frames → 637 frames, 24fps → 96fps (use with `--target-fps 60` to re-encode at 60)
- **Speed**: Fast — the RIFE model is lightweight, processes 1080p pairs in ~0.1s each

### Spatial Upscaling (Real-ESRGAN)
- **Purpose**: Upscale resolution from 1080p to 4K with AI-enhanced detail
- **How**: Deep residual network (RRDB) that hallucinates high-frequency detail, processing each frame as 512px tiles with overlap
- **Model**: RealESRGAN_x2plus (auto-downloaded on first run, ~64MB)
- **2x**: 1920x1088 → 3840x2176
- **4x**: 1920x1088 → 7680x4352
- **Speed**: Slow — each frame is split into 12 tiles, each processed by the GPU. ~2s per frame at 1080p→4K

## Pipeline Per Video

When both `--interpolate` and `--upscale` are used:

1. **RIFE interpolation** → writes `_interp.mp4` (intermediate)
2. **Real-ESRGAN upscaling** of the interpolated video → writes final `.mp4`
3. Intermediate `_interp.mp4` is deleted

Interpolation is done first because:
- RIFE works best on the original resolution (less noise to confuse optical flow)
- Upscaling after interpolation means the smoother motion is preserved in 4K

## File Structure

```
postprocess.sh           # Shell wrapper — mounts volumes, runs Docker
postprocess.py           # Python pipeline — RIFE + Real-ESRGAN logic
Dockerfile.postprocess   # Docker image definition
rife_model/train_log/    # RIFE v4.25 model weights (downloaded separately)
postprocess_batch1.sh    # Batch script for batch1 videos
```

## Estimated Processing Time (Jetson Thor)

Per 161-frame video at 1920x1088:
- Interpolation (2x): ~30 seconds
- Upscaling (2x): ~10-15 minutes (319 interpolated frames × 12 tiles each)
- Total per video: ~15 minutes
- Full batch of 12 videos: ~3 hours
