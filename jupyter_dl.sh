#!/bin/bash
# Start Jupyter with TensorFlow Docker container

# Use tailscale IP instead of local 127 network
TAILSCALE_IP=$(tailscale ip -4)

# -d detached mode so container runs in background. Now tailscale ip can be used
docker run -d --rm --gpus all \
  -v "$PWD":/tf/notebooks \
  -p 8888:8888 \
  tensorflow/tensorflow:2.17.0-gpu-jupyter \
  bash -c "jupyter notebook --ip=0.0.0.0 --allow-root --NotebookApp.token=''"

# ip=0.0.0.0 -> Listens to all networks, not only 127... network

# Output URL to access Jupyter on tailscale ip adress
echo "You can access Jupyter at: http://$TAILSCALE_IP:8888"
