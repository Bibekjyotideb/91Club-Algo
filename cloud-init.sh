#!/bin/bash
# ============================================
# WinGo Predictor — Oracle Cloud Init Script
# Runs automatically on first boot
# ============================================

set -e

LOG="/var/log/wingo-setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== WinGo Setup Started: $(date) ==="

# --- 1. System Update ---
apt-get update -y
apt-get upgrade -y

# --- 2. Install Docker ---
apt-get install -y docker.io docker-compose git
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# --- 3. Open Port 8000 in OS Firewall ---
iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
netfilter-persistent save

# --- 4. Clone the Repo ---
cd /home/ubuntu
git clone https://github.com/Bibekjyotideb/91Club-Algo.git wingo
chown -R ubuntu:ubuntu wingo

# --- 5. Create .env ---
cat > /home/ubuntu/wingo/.env << 'EOF'
HOST=0.0.0.0
PORT=8000
SEQUENCE_LENGTH=30
CONFIDENCE_THRESHOLD=0.55
LEARNING_RATE=0.001
GAME_INTERVAL=3
PHONE_NUMBER=
PASSWORD=
EOF

# --- 6. Create persistent data dirs ---
mkdir -p /home/ubuntu/wingo-data
mkdir -p /home/ubuntu/wingo-models
chown -R ubuntu:ubuntu /home/ubuntu/wingo-data /home/ubuntu/wingo-models

# --- 7. Build Docker Image ---
cd /home/ubuntu/wingo
docker build -t wingo-predictor .

# --- 8. Run Container (auto-restart = 24/7) ---
docker run -d \
  --name wingo \
  --restart=always \
  -p 8000:8000 \
  -v /home/ubuntu/wingo-data:/app/data \
  -v /home/ubuntu/wingo-models:/app/model/checkpoints \
  --env-file /home/ubuntu/wingo/.env \
  wingo-predictor

echo "=== WinGo Setup Complete: $(date) ==="
echo "=== Dashboard: http://$(curl -s ifconfig.me):8000 ==="
