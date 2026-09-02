# Production Dockerfile for Real-Time Face Recognition & Attendance Monitoring System
# Designed for NVIDIA RTX GPU Servers & NVIDIA Jetson Orin/Xavier Edge Devices

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ=Asia/Kolkata
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# Install system dependencies & GStreamer libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN echo '/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib' > /etc/ld.so.conf.d/nvidia_cudnn.conf && \
    echo '/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib' >> /etc/ld.so.conf.d/nvidia_cudnn.conf && \
    echo '/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib' >> /etc/ld.so.conf.d/nvidia_cudnn.conf && \
    ldconfig

# Copy application source code
COPY . .

# Expose Port 9001
EXPOSE 9001

CMD ["python3", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "9001"]
