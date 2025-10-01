#!/bin/bash
# === CONFIG ===
GHCR_USER="anweshaghosh07"                
IMAGE_NAME="green-ai-tracker"            
IMAGE_TAG="latest"
EC2_USER="ubuntu"                        
EC2_HOST="13.233.94.174"            
PEM_PATH="C:\Users\anwes\Downloads\key.pem"                 
PROJECT_DIR="/home/ubuntu/green-ai-tracker"  

# === STEP 1: Commit & Push Code/Data ===
echo "[LOCAL] Remember: run 'git push' and 'dvc push' before this script if you updated data/models."

# === STEP 2: Build & Push Image (only if code changed) ===
echo "[LOCAL] Building Docker image..."
docker build -t ghcr.io/$GHCR_USER/$IMAGE_NAME:$IMAGE_TAG .

echo "[LOCAL] Logging in to GHCR..."
echo $GHCR_PAT | docker login ghcr.io -u $GHCR_USER --password-stdin

echo "[LOCAL] Pushing Docker image..."
docker push ghcr.io/$GHCR_USER/$IMAGE_NAME:$IMAGE_TAG

# === STEP 3: Deploy on EC2 ===
echo "[DEPLOY] Connecting to EC2 ($EC2_HOST)..."
ssh -i $PEM_PATH $EC2_USER@$EC2_HOST << EOF
    set -e
    cd $PROJECT_DIR

    echo "[EC2] Pulling latest repo..."
    git pull origin main

    echo "[EC2] Pulling latest data/models from DVC..."
    dvc pull || true

    echo "[EC2] Pulling latest Docker image..."
    docker pull ghcr.io/$GHCR_USER/$IMAGE_NAME:$IMAGE_TAG

    echo "[EC2] Stopping old container..."
    docker stop greenai || true
    docker rm greenai || true

    echo "[EC2] Starting new container with volume mounts..."
    docker run -d --name greenai -p 80:8501 \
        -v $PROJECT_DIR/data:/app/data \
        -v $PROJECT_DIR/models:/app/models \
        ghcr.io/$GHCR_USER/$IMAGE_NAME:$IMAGE_TAG

    echo "[EC2] Deployment finished. App should be live at http://$EC2_HOST!"
EOF