#!/bin/bash
set -e

# If no args, drop into shell for interactive use
if [ $# -eq 0 ]; then
    echo "LTX-2 Video Generation Container"
    echo "Usage: docker run ... ltx2 --prompt 'your prompt here'"
    echo ""
    echo "Options:"
    echo "  --prompt TEXT          Plain natural language prompt (required)"
    echo "  --lora REPO_ID        HF LoRA repo ID"
    echo "  --lora-weight-name F  Specific weight file in LoRA repo"
    echo "  --lora-scale FLOAT    LoRA scale 0.0-1.0 (default: 1.0)"
    echo "  --no-fp8              Disable FP8 quantization"
    echo "  --steps N             Inference steps (default: 8 for distilled)"
    echo "  --vae-chunk-frames N  VAE decode chunk size (auto if not set)"
    echo "  --output PATH         Output path (default: video.mp4)"
    echo ""
    exec /bin/bash
fi

if [ "$1" = "decode" ]; then
    shift
    exec python3 /app/decode_latents.py "$@"
else
    exec python3 /app/generate.py "$@"
fi
