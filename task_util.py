import lm_eval
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
import torch
import torch.nn as nn
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import re
import os
import numpy as np
from tqdm import tqdm
from typing import Optional, List
import random
import lm_eval
import numpy as np
import math
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import copy
import gc
import torch
import inspect
import socket
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def clean_and_report_cuda_tensors(context_label="未命名"):
    print(f"\n===== 🚀【CUDA 检查开始】[{context_label}] =====")
    
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()

    # 记录当前仍在 CUDA 的 Tensor
    cuda_tensors = []
    total_size_mb = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    size_mb = obj.element_size() * obj.nelement() / 1024**2
                    total_size_mb += size_mb
                    cuda_tensors.append((type(obj), tuple(obj.size()), size_mb))

        except Exception:
            continue

    if cuda_tensors:
        print(f"🔍 找到 {len(cuda_tensors)} 个仍驻留在 CUDA 上的张量:")
        for obj_type, shape, size in cuda_tensors:
            print(f"  - 类型: {obj_type.__name__:<20} | 尺寸: {shape} | 显存: {size:.2f} MB")
    else:
        print("✅ 没有发现任何驻留在 CUDA 上的张量，显存清理成功。")

    print(f"🧠 总占用 CUDA 显存（非缓存）: {total_size_mb:.2f} MB")
    print(f"===== ✅【CUDA 检查结束】[{context_label}] =====\n")
    
def list_tensors_on_cuda():
    print("="*50)
    print("🔍 当前仍驻留在 CUDA 上的张量信息：")
    total_mem = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    size = obj.element_size() * obj.nelement() / 1024**2  # MB
                    total_mem += size
                    print(f"- 类型: {type(obj)}, 大小: {obj.size()}, 占用显存: {size:.2f} MB")
        except Exception as e:
            pass
    print(f"\n🧠 总共占用 CUDA 显存（非缓存部分）: {total_mem:.2f} MB")
    print("="*50)

def extract_task_groups(log_file_path = 'logs11.log'):
    task_groups = []  # 存放任务组的列表
    start_line = False  # 标记是否已经找到目标行
    with open(log_file_path, 'r') as file:
        for line in file:
            # 检查是否为开始的分隔行
            if 'Group' in line and 'Config Location' in line:
                start_line = True
                print('Group Find')
                continue  # 跳过该行，直接检查下一行
            
            # 从目标行的下一行开始读取
            if start_line:
                # 如果遇到空行则停止
                if not line.strip():
                    break
                
                # 使用正则提取任务组名称（即前面的部分）
                match = re.match(r'\| *(\S+)', line.strip())  # 匹配从|开始的第一个非空白字符串
                if match:
                    group = match.group(1).strip().split('|')
                    task_groups.append(group[0])  # 提取任务组名称并添加到列表中

    return task_groups

def extract_task_dict(log_file_path ,task_groups):
    task_dict = {}  # 存放每个task_group对应的任务列表
    start_line_task = False  # 标记是否找到任务的部分
    for task_group in task_groups:
        # 如果任务组不存在，则初始化为空列表
        if task_group not in task_dict:
            task_dict[task_group] = []
            
    # 读取日志文件
    with open(log_file_path, 'r') as file:
        for line in file:
            # 2. 查找任务（task）部分
            if 'Task' in line and 'Config Location' in line and 'Output Type' in line:
                start_line_task = True
                continue  # 跳过该行，继续处理下一行
            
            if start_line_task:
                if not line.strip():  # 如果遇到空行则停止
                    break
                # 使用正则提取任务名称（即前面的部分）
                match = re.match(r'\| *(\S+)', line.strip())  # 匹配任务名称
                if match:
                    task_name = match.group(1).strip()  # 提取任务名称并去掉空格
                    # 遍历所有task_group并检查任务名称是否包含task_group中每一部分
                    for task_group in task_groups:
                        # 分割task_group为多个部分
                        task_group_parts = task_group.split('_')
                        # 判断任务名称是否包含所有的task_group_parts中的元素
                        if all(part in task_name for part in task_group_parts):
                            task_dict[task_group].append(task_name)  # 将任务名称添加到对应的task_group列表中

    return task_dict

def compute_frequency_variance(frequency_list):
    """
    Compute the variance of frequencies for each expert in the given frequency list.
    
    Args:
        frequency_list (List[float]): The list of frequencies for each expert in the layer.
    
    Returns:
        List[float]: Variance of frequencies for each expert.
    """
    mean_frequency = np.mean(frequency_list)
    variance_list = [(freq - mean_frequency) ** 2 for freq in frequency_list]
    return variance_list

def compute_similarity_between_layers(layer_variance_1, layer_variance_2):
    """
    Compute the cosine similarity between the frequency variance of two layers.
    
    Args:
        layer_variance_1 (List[float]): Variance of frequencies for each expert in the first layer.
        layer_variance_2 (List[float]): Variance of frequencies for each expert in the second layer.
    
    Returns:
        float: Cosine similarity between the two variance lists.
    """
    cos_sim = cosine_similarity(
        np.array(layer_variance_1).reshape(1, -1),
        np.array(layer_variance_2).reshape(1, -1)
    )[0][0]
    return cos_sim

def map_to_range(data,name, new_min=0.1, new_max=0.9):
    if name == []:
        # 定义函数，将每个小列表映射到[0.1, 0.9]区间
        result = []
        names = []
        for gate, values in data.items():
            old_min = min(values)
            old_max = max(values)
            # 遍历每个值进行归一化
            normalized_values = [
                0 if math.isnan(value) else value
                for value in values
            ]
            result.append(normalized_values)
            names.append(gate)
    else:
        result = [[] for ii in name]
        names = name
        for namem in names:
            vales = data[namem]
            old_min = min(values)
            old_max = max(values)
            # 遍历每个值进行归一化
            normalized_values = [
                0 if math.isnan(value) else value
                for value in values
            ]
            result.append(normalized_values)
    
    return result,names

def get_model_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    # Convert bytes to megabytes (MB)
    total_size_mb = total_size / (1024 ** 3)
    return total_size_mb

def wait_for_port(ip, port, timeout=30.0, logger=logger):
    """
    Wait for a specific IP and port to become available (open for TCP connection).

    Args:
        ip (str): The IP address to check.
        port (int): The port to check.
        timeout (float): Max time to wait (in seconds).
        logger (logging.Logger, optional): Optional logger for debug output.

    Returns:
        bool: True if port is open before timeout, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)  # 1 second timeout
                result = sock.connect_ex((ip, port))
                if result == 0:
                    if logger:
                        logger.debug(f"Port {port} at {ip} is open.")
                    return True
                else:
                    if logger:
                        logger.debug(f"Port {port} at {ip} not open yet (code {result}).")
        except Exception as e:
            if logger:
                logger.warning(f"Error checking port {port} at {ip}: {e}")
        time.sleep(0.5)

    if logger:
        logger.warning(f"Timeout waiting for port {port} at {ip}.")
    return False
    
def safe_model_clone(model):
    # 保存当前模型的 state_dict
    state_dict = copy.deepcopy(model.state_dict())  # deepcopy here is OK
    # 新建同结构模型
    new_model = type(model)(model.config)
    new_model.load_state_dict(state_dict)
    return new_model