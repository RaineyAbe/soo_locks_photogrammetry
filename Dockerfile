# Use latest micromamba with linux/amd64 platform as base image
FROM --platform=linux/amd64 mambaorg/micromamba:latest

# Set the working 
WORKDIR /app

# Copy environment file to container
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Install dependencies using micromamba
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Install system dependencies required by opencv-python
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Copy code into container
COPY . .

ENTRYPOINT ["bash"]
