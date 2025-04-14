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
