#!/usr/bin/env bash
# ==============================================================================
# Setup script for Edge Client on AWS EC2 (Ubuntu / Debian, 4GB RAM)
# ==============================================================================
set -e

echo "🚀 Setting up AI Monk Edge Client on EC2..."

# 1. Update & install system dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-dev ffmpeg libsm6 libxext6 libgl1 libglib2.0-0

# 2. Install lightweight edge Python requirements (Total size < 150MB)
pip3 install --upgrade pip
pip3 install onnxruntime opencv-python-headless numpy requests urllib3

# 3. Create models directory
mkdir -p models/ultra_light

echo "✅ EC2 Edge Client environment ready!"
echo ""
echo "👉 To run the Edge Client on a phone video:"
echo "   python3 edge_client.py --video your_phone_video.mp4 --server https://49.206.228.75:9001 --output annotated_output.mp4"
