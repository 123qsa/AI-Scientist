"""
多 GPU 实验管理器
为 4x RTX 4090 配置优化
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
from typing import Dict, List, Optional
from pathlib import Path


class MultiGPUExperimentManager:
    """多GPU实验管理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.world_size = config.get('world_size', torch.cuda.device_count())
        self.master_addr = config.get('master_addr', 'localhost')
        self.master_port = config.get('master_port', '29500')

    def setup_distributed(self, rank: int):
        """设置分布式环境"""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)

        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=self.world_size
        )

        torch.cuda.set_device(rank)

    def cleanup(self):
        """清理分布式环境"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def run_experiment(self, rank: int, experiment_fn, *args, **kwargs):
        """在指定GPU上运行实验"""
        self.setup_distributed(rank)

        try:
            # 调用实验函数
            result = experiment_fn(rank, self.world_size, *args, **kwargs)

            # 只在主进程保存结果
            if rank == 0:
                print(f"✓ Experiment completed on GPU {rank}")

            return result

        except Exception as e:
            print(f"❌ Error on GPU {rank}: {e}")
            raise

        finally:
            self.cleanup()

    def launch(self, experiment_fn, *args, **kwargs) -> List:
        """启动多GPU训练"""
        print(f"🚀 Launching distributed training on {self.world_size} GPUs")

        # 使用spawn启动多进程
        mp.spawn(
            self.run_experiment,
            args=(experiment_fn, *args),
            nprocs=self.world_size,
            join=True
        )

    def get_optimal_batch_size(self, model_name: str) -> int:
        """
        根据模型获取最优batch size
        针对4090 24GB优化
        """
        batch_sizes = {
            'resnet50': 64,      # 4x64 = 256
            'resnet101': 32,     # 4x32 = 128
            'vit_base': 32,      # 4x32 = 128
            'vit_large': 16,     # 4x16 = 64
            'swin_base': 24,     # 4x24 = 96
            'faster_rcnn': 2,    # 检测占用大
            'mask_rcnn': 1,      # 实例分割
            'deeplabv3': 4,      # 语义分割
        }
        return batch_sizes.get(model_name, 32)


class GPUAllocator:
    """GPU资源分配器"""

    def __init__(self, total_gpus: int = 4):
        self.total_gpus = total_gpus
        self.gpu_status = {i: 'free' for i in range(total_gpus)}

    def allocate(self, num_gpus: int = 1) -> Optional[List[int]]:
        """分配GPU"""
        free_gpus = [i for i, status in self.gpu_status.items() if status == 'free']

        if len(free_gpus) < num_gpus:
            return None

        allocated = free_gpus[:num_gpus]
        for gpu in allocated:
            self.gpu_status[gpu] = 'busy'

        return allocated

    def release(self, gpus: List[int]):
        """释放GPU"""
        for gpu in gpus:
            if gpu in self.gpu_status:
                self.gpu_status[gpu] = 'free'

    def get_status(self) -> Dict:
        """获取所有GPU状态"""
        return self.gpu_status.copy()

    def wait_for_gpus(self, num_gpus: int = 1, timeout: int = 3600) -> List[int]:
        """等待可用的GPU"""
        print(f"⏳ Waiting for {num_gpus} free GPU(s)...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            allocated = self.allocate(num_gpus)
            if allocated:
                print(f"✓ Allocated GPUs: {allocated}")
                return allocated
            time.sleep(10)

        raise TimeoutError(f"Timeout waiting for {num_gpus} GPUs")


class ParallelExperimentRunner:
    """并行实验运行器 - 同时运行多个实验"""

    def __init__(self, gpu_allocator: GPUAllocator):
        self.gpu_allocator = gpu_allocator
        self.experiments = []

    def add_experiment(self, name: str, config: Dict, num_gpus: int = 1):
        """添加实验到队列"""
        self.experiments.append({
            'name': name,
            'config': config,
            'num_gpus': num_gpus,
            'status': 'pending'
        })

    def run_parallel(self, max_concurrent: int = 2):
        """
        并行运行实验
        例如: 4张卡，每张实验用2卡，可同时运行2个实验
        """
        import concurrent.futures

        completed = 0
        total = len(self.experiments)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []

            for exp in self.experiments:
                # 等待GPU可用
                gpus = self.gpu_allocator.wait_for_gpus(exp['num_gpus'])

                # 提交实验
                future = executor.submit(
                    self._run_single_experiment,
                    exp,
                    gpus
                )
                futures.append((future, exp, gpus))

            # 等待完成
            for future, exp, gpus in futures:
                try:
                    result = future.result()
                    exp['status'] = 'completed'
                    completed += 1
                    print(f"✓ {exp['name']} completed ({completed}/{total})")
                except Exception as e:
                    exp['status'] = 'failed'
                    print(f"❌ {exp['name']} failed: {e}")
                finally:
                    self.gpu_allocator.release(gpus)

    def _run_single_experiment(self, experiment: Dict, gpus: List[int]):
        """运行单个实验"""
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

        # 这里调用实际的实验代码
        print(f"Running {experiment['name']} on GPUs {gpus}")

        # 模拟实验运行
        time.sleep(5)

        return {'status': 'success', 'gpus': gpus}


def benchmark_gpus():
    """GPU基准测试"""
    print("🧪 Benchmarking GPUs...")

    num_gpus = torch.cuda.device_count()
    results = {}

    for i in range(num_gpus):
        torch.cuda.set_device(i)

        # 测试显存带宽
        size = 1024 * 1024 * 100  # 100MB
        a = torch.randn(size, device=f'cuda:{i}')
        b = torch.randn(size, device=f'cuda:{i}')

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):
            c = a + b

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # 测试计算能力
        d = torch.randn(1000, 1000, device=f'cuda:{i}')
        e = torch.randn(1000, 1000, device=f'cuda:{i}')

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):
            f = torch.matmul(d, e)

        torch.cuda.synchronize()
        compute_time = time.time() - start

        results[f'GPU_{i}'] = {
            'memory_bw_time': elapsed,
            'compute_time': compute_time,
            'name': torch.cuda.get_device_name(i)
        }

        del a, b, c, d, e, f
        torch.cuda.empty_cache()

    print("\nBenchmark Results:")
    for gpu, metrics in results.items():
        print(f"  {gpu}: {metrics['name']}")
        print(f"    Memory Bandwidth Test: {metrics['memory_bw_time']:.3f}s")
        print(f"    Compute Test: {metrics['compute_time']:.3f}s")

    return results


if __name__ == "__main__":
    # 测试
    print(f"Detected {torch.cuda.device_count()} GPUs")

    if torch.cuda.is_available():
        benchmark_gpus()

        # 测试GPU分配器
        allocator = GPUAllocator()
        print(f"\nGPU Status: {allocator.get_status()}")

        gpus = allocator.allocate(2)
        print(f"Allocated: {gpus}")
        print(f"Status after alloc: {allocator.get_status()}")

        allocator.release(gpus)
        print(f"Status after release: {allocator.get_status()}")
