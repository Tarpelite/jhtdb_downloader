#!/bin/bash
# start_worker.sh
WORKER_ID=$1
MASTER_IP="94.74.65.162"


# 创建并挂载tmpfs
mkdir -p /mnt/tmpfs
mount -t tmpfs -o size=32G tmpfs /mnt/tmpfs

# 安装依赖
pip install -r requirements.txt

# 启动worker
python worker_node.py \
    --worker-id $WORKER_ID \
    --total-workers 10 \
    --master-url "http://$MASTER_IP:5000" \
    --redis-host $MASTER_IP \
    --redis-port 6379 \
    --token "your-token-here" \
    --obs-bucket "obsobs" \
    --max-workers 16

