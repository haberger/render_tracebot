FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04 
ARG PYTHON_VERSION=3.8
#----------------------------------
# ARG USER_ID
# ARG GROUP_ID
# RUN addgroup --gid $GROUP_ID user
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
#---------------------------------

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxi6 \
    libxrender1 \
    fontconfig

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    open3d==0.13.0\
    pyrender==0.1.45 \
    opencv-python==4.7.0.72 \
    trimesh==3.21.3

# USER user
WORKDIR /workspace