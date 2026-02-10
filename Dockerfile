# LTX-2 Video Generation on Jetson AGX Thor
# Base: NGC PyTorch 26.01 (JetPack 7.0, CUDA 13.0, sm_110 Blackwell)
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Neutralize NGC pip constraint file and config
RUN pip config unset global.constraint 2>/dev/null || true && \
    rm -f /etc/pip/constraint.txt && \
    rm -f /etc/pip/pip.conf && \
    rm -f /etc/pip/pip.ini

# ffmpeg for audio muxing into MP4
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install diffusers from git main (needed for LTX2Pipeline + connectors)
# NEVER reinstall torch — it's built for CUDA 13 + sm_110
RUN PIP_CONSTRAINT="" pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git@5bf248ddd8796b4f4958559429071a28f9b2dd3a \
    transformers>=4.48.0 \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    peft \
    imageio \
    imageio-ffmpeg \
    huggingface_hub[cli]

# Copy application files
WORKDIR /app
COPY generate.py decode_latents.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# HF cache mount point — offline mode skips hub checks, uses local cache only
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_OFFLINE=1
VOLUME ["/root/.cache/huggingface"]

ENTRYPOINT ["./entrypoint.sh"]
