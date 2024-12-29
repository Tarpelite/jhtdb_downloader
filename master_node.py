import os
import json
import time
import logging
import redis
import numpy as np
from datetime import datetime
from flask import Flask, jsonify
from dataclasses import dataclass
from typing import Dict, List
import threading

app = Flask(__name__)

@dataclass
class MasterConfig:
    """Master节点配置"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    total_workers: int = 10
    web_port: int = 5000
    task_check_interval: int = 60  # 检查任务状态的间隔(秒)

class DownloadMaster:
    def __init__(self, config: MasterConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        self.setup_logging()
        self.total_tasks = self.calculate_total_tasks()
        
    def setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )

    def initialize_progress_tracking(self):
        """初始化进度追踪"""
        try:
            # 使用Redis事务确保原子性
            pipe = self.redis_client.pipeline()
            pipe.set('total_tasks', self.total_tasks)
            pipe.set('completed_tasks', 0)
            
            # 初始化每个worker的状态
            for worker_id in range(1, self.config.total_workers + 1):
                pipe.hset('worker_status', f'worker_{worker_id}', 'offline')
                pipe.hset('worker_progress', f'worker_{worker_id}', '0')
                pipe.hset('worker_last_heartbeat', f'worker_{worker_id}', str(int(time.time())))
            
            pipe.execute()
        except Exception as e:
            logging.error(f"Failed to initialize progress tracking: {e}")
            raise

    def calculate_total_tasks(self) -> int:
        """计算总任务数"""
        x_blocks = 1024 // 512  # 2
        y_blocks = 1024 // 512  # 2
        z_blocks = 1024 // 2    # 512
        timesteps = 1024 // 8 # 128
        fields = 4  # u(3) + b(3) + p(1)
        return x_blocks * y_blocks * z_blocks * timesteps * fields

    def check_worker_status(self):
        """检查worker状态并更新"""
        while True:
            current_time = int(time.time())
            for worker_id in range(1, self.config.total_workers + 1):
                worker_key = f'worker_{worker_id}'
                last_heartbeat = int(float(self.redis_client.hget('worker_last_heartbeat', worker_key) or 0))
                
                # 如果超过5分钟没有心跳，标记为离线
                if current_time - last_heartbeat > 300:
                    self.redis_client.hset('worker_status', worker_key, 'offline')
                    logging.warning(f"Worker {worker_id} appears to be offline")
            
            time.sleep(self.config.task_check_interval)

    def start(self):
        """启动master服务"""
        self.initialize_progress_tracking()
        
        # 启动状态检查线程
        status_thread = threading.Thread(target=self.check_worker_status, daemon=True)
        status_thread.start()
        
        # 启动Web服务
        app.run(host='0.0.0.0', port=self.config.web_port)

# Flask路由
@app.route('/status')
def get_status():
    """获取当前下载状态"""
    redis_client = app.config['redis_client']
    
    total_tasks = int(redis_client.get('total_tasks') or 0)
    completed_tasks = int(redis_client.get('completed_tasks') or 0)
    
    worker_status = redis_client.hgetall('worker_status')
    worker_progress = redis_client.hgetall('worker_progress')
    
    current_time = int(time.time())
    worker_details = []
    
    for worker_id in range(1, app.config['total_workers'] + 1):
        worker_key = f'worker_{worker_id}'
        last_heartbeat = int(float(redis_client.hget('worker_last_heartbeat', worker_key) or 0))
        
        worker_details.append({
            'worker_id': worker_id,
            'status': worker_status.get(worker_key, 'unknown'),
            'progress': worker_progress.get(worker_key, '0'),
            'last_heartbeat': f"{current_time - last_heartbeat}s ago"
        })

    return jsonify({
        'total_progress': f"{(completed_tasks / total_tasks * 100):.2f}%" if total_tasks > 0 else "0%",
        'completed_tasks': completed_tasks,
        'total_tasks': total_tasks,
        'workers': worker_details,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/worker/<int:worker_id>/heartbeat', methods=['POST'])
def update_worker_heartbeat(worker_id):
    """更新worker心跳"""
    redis_client = app.config['redis_client']
    worker_key = f'worker_{worker_id}'
    
    redis_client.hset('worker_last_heartbeat', worker_key, str(int(time.time())))
    redis_client.hset('worker_status', worker_key, 'online')
    
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    config = MasterConfig(
        redis_host='localhost',
        redis_port=6379,
        total_workers=6
    )
    
    master = DownloadMaster(config)
    
    # 将redis客户端添加到Flask配置中
    app.config['redis_client'] = master.redis_client
    app.config['total_workers'] = config.total_workers
    
    master.start()
