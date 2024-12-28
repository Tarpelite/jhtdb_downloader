#!/bin/bash
# start_master.sh
redis-server &
python master_node.py \
    --redis-port 6379 \
    --total-workers 10 \
    --web-port 5000