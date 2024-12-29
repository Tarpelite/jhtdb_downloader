import concurrent.futures
import os
import json
import time
import logging
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import List, Dict, Optional
import zeep
import pickle
import struct
import redis
import requests
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import random
from pathlib import Path
from tqdm import tqdm


def download_chunk(task: Dict):
    """下载单个数据块"""
    try:
        # 在每个进程中创建新的client
        client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
        
        # 添加随机延迟避免同时请求
        time.sleep(random.uniform(0, 0.1))
        
        result = client.service.GetAnyCutoutWeb(
            "cn.edu.pku.shanghang-22b6171a",  # 这里需要从外部传入token
            "mhd1024",
            task['field'],
            task['timestep'],
            task['x_start'], task['y_start'], task['z_start'],
            task['x_end'], task['y_end'], task['z_end'],
            task['x_step'], task['y_step'], task['z_step'],
            0, ""
        )
        
        # 解析数据
        c = 1 if task['field'] == 'p' else 3
        base64_len = int(512 * 512 * 2 * c)
        base64_format = '<' + str(base64_len) + 'f'
        data = struct.unpack(base64_format, result)
        data = np.array(data).reshape((512, 512, 2, c))
        
        # 保存到临时文件系统
        save_name = f"f_{task['field']}_t_{task['timestep']}_z_{task['z_start']}.npy"
        local_path = f'/mnt/tmpfs/{save_name}'
        np.save(local_path, data)
        
        # 上传到OBS
        remote_path = f"obs://jhtdb/worker_{task['worker_id']}/{save_name}"
        os.system(f"obsutil cp {local_path} {remote_path} >log.txt 2>&1")
        
        # 清理临时文件
        os.remove(local_path)
        
        return True
        
    except Exception as e:
        logging.error(f"Error in download_chunk: {str(e)}")
        return False


@dataclass
class WorkerConfig:
    """Worker配置类"""
    dataset_name: str = "mhd1024"
    worker_id: int = 1
    total_workers: int = 6
    token: str = "your-token"
    max_workers: int = 48  # 每个节点的并发数
    retry_limit: int = 3
    obs_bucket: str = "your-bucket"
    tmpfs_path: str = "/mnt/tmpfs"
    master_url: str = "http://master:5000"
    redis_host: str = "localhost"
    redis_port: int = 6379
    heartbeat_interval: int = 60
    rate_limit_sleep: float = 0.1

class JHTDBWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.setup_logging()
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        self.setup_directories()
        
    def setup_logging(self):
        """配置日志系统"""
        log_file = f"worker_{self.config.worker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def setup_directories(self):
        """创建必要的目录"""
        Path(self.config.tmpfs_path).mkdir(parents=True, exist_ok=True)
        
    def send_heartbeat(self):
        """发送心跳到master"""
        while True:
            try:
                response = requests.post(
                    f"{self.config.master_url}/worker/{self.config.worker_id}/heartbeat"
                )
                if response.status_code != 200:
                    logging.error(f"Failed to send heartbeat: {response.status_code}")
            except Exception as e:
                logging.error(f"Heartbeat error: {e}")
            
            time.sleep(self.config.heartbeat_interval)
            
    def update_progress(self, completed_tasks: int):
        """更新进度"""
        try:
            self.redis_client.incr('completed_tasks')
            self.redis_client.hset(
                'worker_progress',
                f'worker_{self.config.worker_id}',
                str(completed_tasks)
            )
        except Exception as e:
            logging.error(f"Failed to update progress: {e}")
            
    def upload_to_obs(self, local_path: str, remote_path: str):
        """上传文件到OBS"""
        retry_count = 0
        while retry_count < self.config.retry_limit:
            try:
                os.system(f"./obsutil cp {local_path} {remote_path} >log.txt 2>&1")
                return True
            except Exception as e:
                retry_count += 1
                logging.error(f"OBS upload failed: {e}")
                time.sleep(2 ** retry_count)
        return False
        
    def generate_tasks(self):
        """生成当前worker的下载任务列表，使用1-based索引"""
        tasks = []
        worker_id = self.config.worker_id
        total_workers = self.config.total_workers
        
        for t in range(0, 1024, 8):  # 时间步长
            if t % total_workers == (worker_id - 1):  # 按worker_id分配时间步长
                for field in ['u', 'a', 'b', 'p']:
                    for z in range(0, 1024, 2):
                        for x in range(0, 1024, 512):
                            for y in range(0, 1024, 512):
                                task = {
                                    'timestep': t + 1,  # 1-based indexing
                                    'field': field,
                                    'x_start': x + 1,   # 1-based indexing
                                    'y_start': y + 1,   # 1-based indexing
                                    'z_start': z + 1,   # 1-based indexing
                                    'x_end': min(x + 512, 1024),  # 确保不超过范围
                                    'y_end': min(y + 512, 1024),
                                    'z_end': min(z + 2, 1024),
                                    'x_step': 1,
                                    'y_step': 1,
                                    'z_step': 1,
                                    'worker_id': self.config.worker_id
                                }
                                tasks.append(task)
        
        return tasks
    
    def run(self):
        """运行worker"""
        # 启动心跳线程
        import threading
        heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        # 生成任务列表
        tasks = self.generate_tasks()
        for task in tasks:
            task['worker_id'] = self.config.worker_id  # 添加worker_id到任务中
            
        random.shuffle(tasks)  # 随机化任务顺序
        
        completed_tasks = 0
        failed_tasks = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            with tqdm(total=len(tasks)) as pbar:
                for task in tasks:
                    futures.append(executor.submit(download_chunk, task))
                    
                for future in concurrent.futures.as_completed(futures):
                    try:
                        success = future.result()
                        if success:
                            completed_tasks += 1
                            self.update_progress(completed_tasks)
                        else:
                            failed_tasks.append(task)
                    except Exception as e:
                        logging.error(f"Task failed with error: {e}")
                        failed_tasks.append(task)
                    finally:
                        pbar.update(1)
                        pbar.set_description(f"Progress: {completed_tasks}/{len(tasks)}")

        # 处理失败的任务
        if failed_tasks:
            logging.info(f"Retrying {len(failed_tasks)} failed tasks...")
            retry_tasks = []
            for task in failed_tasks:
                for _ in range(self.config.retry_limit):
                    if download_chunk(task):
                        completed_tasks += 1
                        self.update_progress(completed_tasks)
                        break
                    time.sleep(2)  # 重试前等待
                else:
                    retry_tasks.append(task)
            
            if retry_tasks:
                logging.error(f"Failed to download {len(retry_tasks)} tasks after all retries")
                # 保存失败的任务列表以便后续处理
                failed_tasks_file = f"failed_tasks_worker_{self.config.worker_id}.json"
                with open(failed_tasks_file, 'w') as f:
                    json.dump(retry_tasks, f)

    def cleanup(self):
        """清理临时文件和资源"""
        try:
            # 清理临时文件夹
            for file in os.listdir(self.config.tmpfs_path):
                try:
                    os.remove(os.path.join(self.config.tmpfs_path, file))
                except Exception as e:
                    logging.error(f"Failed to remove temp file {file}: {e}")
            
            # 关闭Redis连接
            if self.redis_client:
                self.redis_client.close()
            
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='JHTDB Worker Node')
    parser.add_argument('--worker-id', type=int, required=True, help='Worker ID')
    parser.add_argument('--total-workers', type=int, default=10, help='Total number of workers')
    parser.add_argument('--master-url', type=str, required=True, help='Master node URL')
    parser.add_argument('--redis-host', type=str, required=True, help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    parser.add_argument('--token', type=str, required=True, help='JHTDB API token')
    parser.add_argument('--obs-bucket', type=str, required=True, help='OBS bucket name')
    parser.add_argument('--max-workers', type=int, default=48, help='Max concurrent workers')
    
    args = parser.parse_args()
    
    config = WorkerConfig(
        worker_id=args.worker_id,
        total_workers=args.total_workers,
        master_url=args.master_url,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        token=args.token,
        obs_bucket=args.obs_bucket,
        max_workers=args.max_workers
    )
    
    worker = JHTDBWorker(config)
    
    try:
        worker.run()
    except KeyboardInterrupt:
        logging.info("Worker interrupted by user")
    except Exception as e:
        logging.error(f"Worker failed with error: {e}")
    finally:
        worker.cleanup()

if __name__ == "__main__":
    main()
