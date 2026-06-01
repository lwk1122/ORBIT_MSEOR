#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="orbit-or-sandbox:latest"
CONTAINER_NAME="orbit-or-sandbox"
# Must match useradd -u in Dockerfile
SANDBOX_UID=1000

echo "Building sandbox image..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "Refreshing sandbox container..."
if docker inspect "$CONTAINER_NAME" &>/dev/null; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --network=none \
    --restart=unless-stopped \
    --memory=2g \
    --cpus=2 \
    --pids-limit=256 \
    --security-opt=no-new-privileges \
    --cap-drop=ALL \
    --read-only \
    --tmpfs "/tmp:size=100m,uid=${SANDBOX_UID},gid=${SANDBOX_UID}" \
    --tmpfs "/workspace/output:size=100m,uid=${SANDBOX_UID},gid=${SANDBOX_UID}" \
    "$IMAGE_NAME" \
    sleep infinity

echo "Container '$CONTAINER_NAME' started with refreshed image."
