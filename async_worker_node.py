import asyncio
import aiohttp
import threading
import os
import json
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import zeep
import pickle
import struct
import redis
import requests
from datetime import datetime
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import deque

class TaskTimeoutError(Exception):
    pass

class DownloadManager:
    def __init__(self, max_workers: int = 48):
        self.max_workers = max_workers
        self.active_tasks = set()
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.task_queue = deque()
        self.task_times = []  # 记录任务完成时间
        self.lock = threading.Lock()
        self._stop = False
        
    def add_task(self, task: Dict):
        self.task_queue.append(task)
    
    def get_average_time(self) -> float:
        """计算平均任务完成时间"""
        if not self.task_times:
            return 60  # 默认60秒
        return sum(self.task_times) / len(self.task_times)
    
    async def download_with_timeout(self, task: Dict):
        """带超时控制的下载任务"""
        timeout = self.get_average_time() * 2  # 使用平均时间的2倍作为超时时间
        
        try:
            start_time = time.time()
            success = await asyncio.wait_for(self._download_chunk(task), timeout)
            
            if success:
                with self.lock:
                    self.task_times.append(time.time() - start_time)
                    if len(self.task_times) > 100:  # 只保留最近100个任务的时间
                        self.task_times.pop(0)
                return True
            return False
            
        except asyncio.TimeoutError:
            logging.warning(f"Task timed out after {timeout:.2f} seconds")
            raise TaskTimeoutError()
        except Exception as e:
            logging.error(f"Download error: {str(e)}")
            return False

    async def _download_chunk(self, task: Dict):
        """异步下载单个数据块"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # 使用 aiohttp 替代 zeep 进行异步请求
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'http://turbulence.pha.jhu.edu/service/turbulence.asmx',
                        data={
                            'token': "cn.edu.pku.shanghang-22b6171a",
                            'dataset': "mhd1024",
                            'field': task['field'],
                            'timestep': task['timestep'],
                            'x_start': task['x_start'],
                            'y_start': task['y_start'],
                            'z_start': task['z_start'],
                            'x_end': task['x_end'],
                            'y_end': task['y_end'],
                            'z_end': task['z_end'],
                            'x_step': task['x_step'],
                            'y_step': task['y_step'],
                            'z_step': task['z_step']
                        }
                    ) as response:
                        result = await response.read()
                
                c = 1 if task['field'] == 'p' else 3
                base64_len = int(512 * 512 * 2 * c)
                base64_format = '<' + str(base64_len) + 'f'
                data = struct.unpack(base64_format, result)
                data = np.array(data).reshape((512, 512, 2, c))
                
                save_name = f"f_{task['field']}_t_{task['timestep']}_x_{task['x_start']}_y_{task['y_start']}_z_{task['z_start']}.npy"
                local_path = f'/mnt/tmpfs/{save_name}'
                np.save(local_path, data)
                
                remote_path = f"obs://mhd1024t128/worker_{task['worker_id']}/{save_name}"
                await asyncio.to_thread(
                    lambda: os.system(f"obsutil cp {local_path} {remote_path} >log.txt 2>&1")
                )
                
                try:
                    os.remove(local_path)
                except:
                    pass
                
                return True
                
            except Exception as e:
                retry_count += 1
                await asyncio.sleep(min(300, (2 ** retry_count) + random.uniform(0, 1)))
                
        return False

    async def worker(self):
        """异步工作器"""
        while not self._stop or self.task_queue:
            if len(self.active_tasks) >= self.max_workers:
                await asyncio.sleep(0.1)
                continue
                
            try:
                task = self.task_queue.popleft()
            except IndexError:
                if not self._stop:
                    await asyncio.sleep(0.1)
                continue
                
            self.active_tasks.add(task['timestep'])
            
            try:
                success = await self.download_with_timeout(task)
                if success:
                    self.completed_tasks.add(task['timestep'])
                else:
                    self.failed_tasks.add(task)
            except TaskTimeoutError:
                self.failed_tasks.add(task)
            finally:
                self.active_tasks.remove(task['timestep'])

    async def run(self, tasks: List[Dict]):
        """运行下载管理器"""
        for task in tasks:
            self.add_task(task)
            
        workers = [self.worker() for _ in range(self.max_workers)]
        with tqdm(total=len(tasks)) as pbar:
            last_completed = 0
            while self.task_queue or self.active_tasks:
                current_completed = len(self.completed_tasks)
                if current_completed > last_completed:
                    pbar.update(current_completed - last_completed)
                    last_completed = current_completed
                await asyncio.sleep(0.1)
                
            pbar.update(len(self.completed_tasks) - last_completed)
        
        self._stop = True
        await asyncio.gather(*workers)
        
        return self.completed_tasks, self.failed_tasks

# 将速率控制移到进程级别
def get_delay():
    """获取当前进程的延迟时间"""
    return 0.1  # 使用固定延迟，避免多进程共享问题

def record_error():
    """记录错误并简单增加延迟"""
    time.sleep(0.5)  # 简单的错误处理策略

def check_completed_tasks(worker_id: int, downsample_rate: int = 8):
    """检查OBS上已经完成的任务"""
    completed_tasks = set()
    
    cmd = f"obsutil ls obs://mhd1024t128/worker_{worker_id}/ -limit=131072 >obs_files.txt 2>&1"
    os.system(cmd)
    
    try:
        with open('obs_files.txt', 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.strip().endswith('.npy'):
                filename = line.strip().split('/')[-1]
                parts = filename.replace('.npy', '').split('_')
                field = parts[1]
                timestep = int(parts[3])
                x_start = int(parts[5])
                y_start = int(parts[7])
                z_start = int(parts[9])
                
                task_id = f"{field}_{timestep}_{x_start}_{y_start}_{z_start}"
                completed_tasks.add(task_id)
                
    except Exception as e:
        logging.error(f"Error reading OBS file list: {e}")
    finally:
        try:
            os.remove('obs_files.txt')
        except:
            pass
            
    return completed_tasks

def is_task_completed(task: Dict, completed_tasks: set) -> bool:
    """检查特定任务是否已完成"""
    task_id = f"{task['field']}_{task['timestep']}_{task['x_start']}_{task['y_start']}_{task['z_start']}"
    return task_id in completed_tasks

def download_chunk(task: Dict):
    """下载单个数据块"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            # 使用进程级别的延迟控制
            time.sleep(get_delay())
            
            client = zeep.Client(
                'http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL',
                transport=zeep.Transport(timeout=30)
            )
            
            result = client.service.GetAnyCutoutWeb(
                "cn.edu.pku.shanghang-22b6171a",
                "mhd1024",
                task['field'],
                task['timestep'],
                task['x_start'], task['y_start'], task['z_start'],
                task['x_end'], task['y_end'], task['z_end'],
                task['x_step'], task['y_step'], task['z_step'],
                0, ""
            )
            
            c = 1 if task['field'] == 'p' else 3
            base64_len = int(512 * 512 * 2 * c)
            base64_format = '<' + str(base64_len) + 'f'
            data = struct.unpack(base64_format, result)
            data = np.array(data).reshape((512, 512, 2, c))
            
            save_name = f"f_{task['field']}_t_{task['timestep']}_x_{task['x_start']}_y_{task['y_start']}_z_{task['z_start']}.npy"
            local_path = f'/mnt/tmpfs/{save_name}'
            np.save(local_path, data)
            
            remote_path = f"obs://mhd1024t128/worker_{task['worker_id']}/{save_name}"
            os.system(f"obsutil cp {local_path} {remote_path} >log.txt 2>&1")
            
            try:
                os.remove(local_path)
            except:
                pass
            
            return True
            
        except Exception as e:
            retry_count += 1
            record_error()
            
            backoff_time = min(300, (2 ** retry_count) + random.uniform(0, 1))
            logging.error(
                f"Error in download_chunk (attempt {retry_count}/{max_retries}): {str(e)}\n"
                f"Retrying in {backoff_time:.2f} seconds..."
            )
            
            time.sleep(backoff_time)
            
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
        self.download_manager = DownloadManager(max_workers=config.max_workers)
        
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
            
    def update_progress(self, completed_tasks: int, total_tasks: int):
        """更新进度"""
        try:
            self.redis_client.incr('completed_tasks')
            self.redis_client.hset(
                'worker_progress',
                f'worker_{self.config.worker_id}',
                f"{completed_tasks}/{total_tasks}"
            )
        except Exception as e:
            logging.error(f"Failed to update progress: {e}")

    def generate_tasks(self):
        """生成当前worker的下载任务列表，使用1-based索引"""
        tasks = []
        worker_id = self.config.worker_id
        total_workers = self.config.total_workers
        downsample_rate = 8 # for downsample timestep t
        
        for t in range(0, 1024, downsample_rate):  # 时间步长
            if (t//downsample_rate) % total_workers == (worker_id - 1):  # 按worker_id分配时间步长
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
            
    async def run(self):
        """运行worker的主函数"""
        heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        completed_tasks = check_completed_tasks(self.config.worker_id)
        logging.info(f"Found {len(completed_tasks)} completed tasks")
        
        tasks = self.generate_tasks()
        remaining_tasks = [task for task in tasks 
                         if not is_task_completed(task, completed_tasks)]
        
        self.total_tasks = len(remaining_tasks)
        logging.info(f"Remaining tasks: {self.total_tasks}")
        random.shuffle(remaining_tasks)
        
        completed_tasks, failed_tasks = await self.download_manager.run(remaining_tasks)
        
        logging.info(f"Completed tasks: {len(completed_tasks)}")
        logging.info(f"Failed tasks: {len(failed_tasks)}")
        
        # 保存失败的任务以便后续重试
        if failed_tasks:
            with open(f'failed_tasks_worker_{self.config.worker_id}.pkl', 'wb') as f:
                pickle.dump(failed_tasks, f)
    
    def cleanup(self):
        """清理临时文件和资源"""
        try:
            for file in os.listdir(self.config.tmpfs_path):
                try:
                    os.remove(os.path.join(self.config.tmpfs_path, file))
                except Exception as e:
                    logging.error(f"Failed to remove temp file {file}: {e}")
            
            if self.redis_client:
                self.redis_client.close()
            
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")

async def main():
  
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
        await worker.run()
    except KeyboardInterrupt:
        logging.info("Worker interrupted by user")
    except Exception as e:
        logging.error(f"Worker failed with error: {e}")
    finally:
        worker.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
