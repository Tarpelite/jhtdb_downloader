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


def check_completed_tasks(worker_id: int, downsample_rate: int = 8):
    """检查OBS上已经完成的任务"""
    completed_tasks = set()
    
    # 列出该worker目录下的所有文件
    cmd = f"obsutil ls obs://mhd1024t128/worker_{worker_id}/ -limit=131072 >obs_files.txt 2>&1"
    os.system(cmd)
    
    try:
        with open('obs_files.txt', 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.strip().endswith('.npy'):
                # 从文件名解析任务信息
                filename = line.strip().split('/')[-1]
                # f_{field}_t_{timestep}_x_{x_start}_y_{y_start}_z_{z_start}.npy
                parts = filename.replace('.npy', '').split('_')
                field = parts[1]
                timestep = int(parts[3])
                x_start = int(parts[5])
                y_start = int(parts[7])
                z_start = int(parts[9])
                
                # 创建任务标识符
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

class AdaptiveRateControl:
    def __init__(self):
        self.error_window = []
        self.window_size = 60  # 60秒的窗口
        self.error_threshold = 5  # 窗口内错误数阈值
        self.base_delay = 0.1  # 基础延迟
        self.max_delay = 5.0  # 最大延迟
        self.current_delay = self.base_delay
        self.lock = threading.Lock()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

    def record_error(self):
        """记录错误并调整延迟"""
        current_time = time.time()
        with self.lock:
            self.consecutive_errors += 1
            self.error_window = [t for t in self.error_window 
                               if current_time - t < self.window_size]
            self.error_window.append(current_time)
            
            if len(self.error_window) > self.error_threshold or \
               self.consecutive_errors >= self.max_consecutive_errors:
                self.current_delay = min(self.current_delay * 1.5, self.max_delay)
                logging.info(f"Increasing delay to {self.current_delay:.2f}s due to errors")

    def record_success(self):
        """记录成功并逐渐恢复"""
        with self.lock:
            self.consecutive_errors = 0
            if not self.error_window:  # 只有在没有最近错误时才减小延迟
                self.current_delay = max(
                    self.base_delay,
                    self.current_delay * 0.95
                )

    def get_delay(self):
        """获取当前延迟时间"""
        return self.current_delay

class AdaptiveWorkerControl:
    def __init__(self, initial_workers: int, max_workers: int):
        self.current_workers = initial_workers
        self.max_workers = max_workers
        self.min_workers = 4
        self.success_streak = 0
        self.error_streak = 0
        self.lock = threading.Lock()

    def adjust_workers(self, success: bool):
        """根据成功/失败调整worker数量"""
        with self.lock:
            if success:
                self.success_streak += 1
                self.error_streak = 0
                if self.success_streak >= 10:  # 连续10次成功后增加worker
                    self.current_workers = min(
                        self.current_workers + 1,
                        self.max_workers
                    )
                    self.success_streak = 0
            else:
                self.error_streak += 1
                self.success_streak = 0
                if self.error_streak >= 2:  # 连续2次错误后减少worker
                    self.current_workers = max(
                        self.current_workers - 2,
                        self.min_workers
                    )
                    self.error_streak = 0
            
            return self.current_workers


def download_chunk(task: Dict, rate_control: AdaptiveRateControl):
    """下载单个数据块"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            # 应用动态延迟
            delay = rate_control.get_delay()
            time.sleep(delay)
            
            # 在每个进程中创建新的client
            client = zeep.Client(
                'http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL',
                transport=zeep.Transport(timeout=30)  # 增加超时时间
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
            
            # 处理数据
            c = 1 if task['field'] == 'p' else 3
            base64_len = int(512 * 512 * 2 * c)
            base64_format = '<' + str(base64_len) + 'f'
            data = struct.unpack(base64_format, result)
            data = np.array(data).reshape((512, 512, 2, c))
            
            # 保存到临时文件系统
            save_name = f"f_{task['field']}_t_{task['timestep']}_x_{task['x_start']}_y_{task['y_start']}_z_{task['z_start']}.npy"
            local_path = f'/mnt/tmpfs/{save_name}'
            np.save(local_path, data)
            
            # 上传到OBS
            remote_path = f"obs://mhd1024t128/worker_{task['worker_id']}/{save_name}"
            os.system(f"obsutil cp {local_path} {remote_path} >log.txt 2>&1")
            
            # 清理临时文件
            os.remove(local_path)
            
            # 记录成功
            rate_control.record_success()
            return True
            
        except Exception as e:
            retry_count += 1
            rate_control.record_error()
            
            # 计算退避时间
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
    
    def run(self):
        """运行worker"""
        # 启动心跳线程
        heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        # 检查已完成的任务
        completed_tasks = check_completed_tasks(self.config.worker_id)
        logging.info(f"Found {len(completed_tasks)} completed tasks")
        
        # 生成任务列表并过滤掉已完成的任务
        tasks = self.generate_tasks()
        tasks = [task for task in tasks 
                if not is_task_completed(task, completed_tasks)]
        
        logging.info(f"Remaining tasks: {len(tasks)}")
        random.shuffle(tasks)  # 随机化任务顺序
        
        # 初始化控制器
        rate_control = AdaptiveRateControl()
        worker_control = AdaptiveWorkerControl(
            initial_workers=8,  # 从较小的值开始
            max_workers=self.config.max_workers
        )
        
        completed_tasks = 0
        failed_tasks = []
        
        while tasks:
            current_workers = worker_control.current_workers
            current_batch = tasks[:current_workers]
            tasks = tasks[current_workers:]
            
            with ProcessPoolExecutor(max_workers=current_workers) as executor:
                futures = []
                with tqdm(total=len(current_batch)) as pbar:
                    for task in current_batch:
                        futures.append(executor.submit(
                            download_chunk,
                            task,
                            rate_control
                        ))
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            success = future.result()
                            if success:
                                completed_tasks += 1
                                self.update_progress(completed_tasks)
                                worker_control.adjust_workers(True)
                            else:
                                failed_tasks.append(task)
                                worker_control.adjust_workers(False)
                        except Exception as e:
                            logging.error(f"Task failed with error: {e}")
                            failed_tasks.append(task)
                            worker_control.adjust_workers(False)
                        finally:
                            pbar.update(1)
                            pbar.set_description(
                                f"Progress: {completed_tasks}/{len(tasks)} "
                                f"Workers: {current_workers}"
                            )
            
            # 每批次后短暂暂停
            time.sleep(1)
        
        # 处理失败的任务
        if failed_tasks:
            self.handle_failed_tasks(failed_tasks)

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
