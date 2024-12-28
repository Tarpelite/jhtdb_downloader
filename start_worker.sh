#!/bin/bash
# start_worker.sh
WORKER_ID=$1
MASTER_IP="94.74.65.162"

# 创建并挂载tmpfs
mkdir -p /mnt/tmpfs
mount -t tmpfs -o size=32G tmpfs /mnt/tmpfs

# 安装python环境
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source /root/miniconda3/bin/activate

# 安装依赖
pip3 install -r requirements.txt

# 启动worker
python worker_node.py \
    --worker-id $WORKER_ID \
    --total-workers 8 \
    --master-url "http://$MASTER_IP:5000" \
    --redis-host $MASTER_IP \
    --redis-port 6379 \
    --token "your-token-here" \
    --obs-bucket "obsobs" \
    --max-workers 16

