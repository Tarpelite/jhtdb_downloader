#!/bin/bash
# start_worker.sh
WORKER_ID=$1
MASTER_IP="192.168.0.162"

# 创建并挂载tmpfs
mkdir -p /mnt/tmpfs
mount -t tmpfs -o size=32G tmpfs /mnt/tmpfs

# 安装python环境
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source /root/miniconda3/bin/activate

# 安装依赖
pip3 install -r requirements.txt

#配置obsutil相关
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz
tar -xzvf obsutil_linux_amd64.tar.gz
chmod 755 obsutil_linux_amd64_5.5.12/obsutil
export PATH=$PATH:/root/jhtdb_downloader/obsutil_linux_amd64_5.5.12/


# 启动worker
python worker_node.py \
    --worker-id $WORKER_ID \
    --total-workers 20 \
    --master-url "http://$MASTER_IP:5000" \
    --redis-host $MASTER_IP \
    --redis-port 6379 \
    --token "cn.edu.pku.shanghang-22b6171a" \
    --obs-bucket "jhtdb" \
    --max-workers 48

