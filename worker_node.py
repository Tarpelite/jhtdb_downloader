import threading
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
from concurrent.futures import as_completed, TimeoutError


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
    """下载单个数据块，带有改进的错误处理"""
    retry_count = 0
    max_retries = 5  # 增加最大重试次数
    initial_backoff = 2  # 初始等待时间（秒）
    max_backoff = 600   # 最大等待时间（秒）
    
    while retry_count < max_retries:
        try:
            # 使用进程级别的延迟控制
            time.sleep(get_delay())
            
            # 创建新的客户端实例，避免复用可能已失效的连接
            client = zeep.Client(
                'http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL',
                transport=zeep.Transport(
                    timeout=60,  # 增加超时时间
                    operation_timeout=60,
                    session=requests.Session()  # 使用新的session
                )
            )
            
            # 在调用API之前添加短暂延迟，避免并发请求过多
            time.sleep(random.uniform(0.1, 0.5))
            
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
            
            # 验证返回的数据
            if not result:
                raise ValueError("Empty response from server")
                
            c = 1 if task['field'] == 'p' else 3
            base64_len = int(512 * 512 * 2 * c)
            base64_format = '<' + str(base64_len) + 'f'
            
            try:
                data = struct.unpack(base64_format, result)
            except struct.error as e:
                raise ValueError(f"Invalid data format: {str(e)}")
                
            data = np.array(data).reshape((512, 512, 2, c))
            
            # 验证数据的有效性
            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("Data contains NaN or Inf values")
            
            save_name = f"f_{task['field']}_t_{task['timestep']}_x_{task['x_start']}_y_{task['y_start']}_z_{task['z_start']}.npy"
            local_path = f'/mnt/tmpfs/{save_name}'
            
            # 保存前先验证本地存储空间
            if os.path.exists(local_path):
                os.remove(local_path)
                
            np.save(local_path, data)
            
            # 验证文件是否成功保存
            if not os.path.exists(local_path):
                raise IOError("Failed to save local file")
                
            remote_path = f"obs://mhd1024t128/worker_{task['worker_id']}/{save_name}"
            
            # 使用obsutil上传文件，带有重试机制
            upload_success = False
            upload_retries = 3
            for upload_attempt in range(upload_retries):
                try:
                    os.system(f"obsutil cp {local_path} {remote_path} >log.txt 2>&1")
                    # 验证上传是否成功（可以通过检查obsutil的返回值）
                    upload_success = True
                    break
                except Exception as e:
                    if upload_attempt < upload_retries - 1:
                        time.sleep(random.uniform(1, 3))
                    else:
                        raise IOError(f"Failed to upload to OBS after {upload_retries} attempts")
            
            # 清理本地文件
            try:
                os.remove(local_path)
            except:
                pass
            
            if upload_success:
                return True
            
        except (zeep.exceptions.Fault, zeep.exceptions.TransportError) as e:
            # SOAP服务相关错误
            retry_count += 1
            backoff_time = min(max_backoff, initial_backoff * (2 ** retry_count) + random.uniform(0, 1))
            
            logging.error(
                f"SOAP service error in download_chunk (attempt {retry_count}/{max_retries}): {str(e)}\n"
                f"Task details: field={task['field']}, timestep={task['timestep']}, "
                f"x={task['x_start']}, y={task['y_start']}, z={task['z_start']}\n"
                f"Retrying in {backoff_time:.2f} seconds..."
            )
            
            time.sleep(backoff_time)
            
        except (requests.exceptions.RequestException, ConnectionError) as e:
            # 网络连接相关错误
            retry_count += 1
            backoff_time = min(max_backoff, initial_backoff * (2 ** retry_count) + random.uniform(0, 1))
            
            logging.error(
                f"Network error in download_chunk (attempt {retry_count}/{max_retries}): {str(e)}\n"
                f"Retrying in {backoff_time:.2f} seconds..."
            )
            
            time.sleep(backoff_time)
            
        except Exception as e:
            # 其他未预期的错误
            retry_count += 1
            backoff_time = min(max_backoff, initial_backoff * (2 ** retry_count) + random.uniform(0, 1))
            
            logging.error(
                f"Unexpected error in download_chunk (attempt {retry_count}/{max_retries}): {str(e)}\n"
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
        self.total_tasks = 0
        self.completed_count = 0
        self.average_download_time = 0  # 用于存储平均下载时间
        self.task_times = []  # 用于存储每个任务的下载时间
        
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
            
    def run(self):
        """运行worker"""
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
        
        batch_size = min(8, self.config.max_workers)  # 开始使用较小的批次
        completed_count = 0
        failed_tasks = []
        
        with tqdm(total=self.total_tasks, desc="Overall Progress") as pbar:
            while remaining_tasks:
                current_batch = remaining_tasks[:batch_size]
                remaining_tasks = remaining_tasks[batch_size:]
                
                futures_completed = False
                retry_count = 0
                max_retries = 3
                
                while not futures_completed and retry_count < max_retries:
                    try:
                        with ProcessPoolExecutor(max_workers=batch_size) as executor:
                            futures = {executor.submit(download_chunk, task): task 
                                    for task in current_batch}
                            
                            timeout = max(300, self.average_download_time * 2 if self.average_download_time > 0 else 300)
                            
                            # 使用as_completed带超时，但捕获异常并继续处理
                            completed_futures = []
                            for future in as_completed(futures, timeout=timeout):
                                completed_futures.append(future)
                                task = futures[future]
                                try:
                                    start_time = time.time()
                                    success = future.result(timeout=30)  # 单个任务的超时时间
                                    end_time = time.time()
                                    
                                    # 更新平均下载时间
                                    self.task_times.append(end_time - start_time)
                                    self.average_download_time = sum(self.task_times[-50:]) / min(len(self.task_times), 50)
                                    
                                    if success:
                                        completed_count += 1
                                        self.update_progress(completed_count, self.total_tasks)
                                        # 如果连续成功，逐渐增加批次大小
                                        if completed_count % 10 == 0:
                                            batch_size = min(batch_size + 2, self.config.max_workers)
                                    else:
                                        failed_tasks.append(task)
                                        # 如果失败，减少批次大小
                                        batch_size = max(4, batch_size - 2)
                                        
                                except TimeoutError:
                                    logging.error(f"Task {task} timed out")
                                    failed_tasks.append(task)
                                    batch_size = max(4, batch_size - 2)
                                except Exception as e:
                                    logging.error(f"Task failed with error: {e}")
                                    failed_tasks.append(task)
                                    batch_size = max(4, batch_size - 2)
                                finally:
                                    pbar.update(1)
                                    pbar.set_description(
                                        f"Completed: {completed_count}/{self.total_tasks} "
                                        f"Workers: {batch_size}"
                                    )
                            
                            # 检查是否所有futures都完成了
                            if len(completed_futures) == len(futures):
                                futures_completed = True
                            else:
                                # 有未完成的futures
                                uncompleted = set(futures.keys()) - set(completed_futures)
                                logging.warning(f"{len(uncompleted)} futures uncompleted in batch")
                                # 将未完成的任务添加回任务队列
                                for future in uncompleted:
                                    task = futures[future]
                                    failed_tasks.append(task)
                                retry_count += 1
                                
                    except concurrent.futures.TimeoutError:
                        logging.error(f"Batch timeout after {timeout} seconds")
                        # 将整个批次添加到失败任务中
                        failed_tasks.extend(current_batch)
                        retry_count += 1
                    except Exception as e:
                        logging.error(f"Batch execution failed: {e}")
                        failed_tasks.extend(current_batch)
                        retry_count += 1
                    
                # 如果重试次数达到上限，记录错误并继续下一个批次
                if retry_count >= max_retries:
                    logging.error(f"Batch failed after {max_retries} retries")
                
                # 批次间暂停，根据成功率动态调整
                success_rate = completed_count / (completed_count + len(failed_tasks)) if completed_count + len(failed_tasks) > 0 else 0
                sleep_time = max(1, 5 * (1 - success_rate))  # 成功率越低，暂停时间越长
                time.sleep(sleep_time)
            
            # 处理失败的任务
            if failed_tasks:
                logging.info(f"Processing {len(failed_tasks)} failed tasks...")
                remaining_tasks.extend(failed_tasks)  # 将失败的任务重新添加到队列
                
        if failed_tasks:
            # 保存失败的任务到文件中
            failed_tasks_file = f"failed_tasks_worker_{self.config.worker_id}.json"
            try:
                with open(failed_tasks_file, 'w') as f:
                    json.dump(failed_tasks, f)
                logging.info(f"Failed tasks saved to {failed_tasks_file}")
            except Exception as e:
                logging.error(f"Failed to save failed tasks: {e}")

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
