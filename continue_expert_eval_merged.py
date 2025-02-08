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

# 初始化一个全局字典来保存捕获的router_logits
captured_outputs = {}
captured_one = {}
layer_count = 0

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

def normalize_data(values):
    min_val = min(values)
    max_val = max(values)
    return [0.01 + (v - min_val) / (max_val - min_val) * (0.99 - 0.01) for v in values]

def _merge_experts_by_usage_frequency_weighting(
        ffn,
        group
):
    """
    Merge experts within a group by usage frequency weighting.
    All experts in a group will share the same parameters after merging.

    Args:
        ffn (PhiMoESparseMoeBlock): The PhiMoE block containing experts.
        group (List[int]): Group labels indicating which group each expert belongs to (length = 16).
    Returns:
        PhiMoESparseMoeBlock: The updated PhiMoE block with merged experts.
    """
    # Convert group and frequency list to tensors for easier processing
    group_tensor = torch.tensor(group, dtype=torch.long)
    
    for label in group_tensor.unique():
        # Find the indices of experts in this group
        expert_indices = torch.where(group_tensor == label)[0]
        
        with torch.no_grad():
            # Bind all experts in the group to the first expert (sharing parameters)
            for expert_idx in expert_indices[1:]:
                ffn.experts[expert_idx] = ffn.experts[expert_indices[0]]
            print(expert_indices[0])
    
    return ffn

def merge_by_groups_with_usage_frequency_weighting(
        model
) :
    """
    Merge experts in the specified layers of the model based on group labels and usage frequencies.

    Args:
        model (PhiMoEForCausalLM): The model instance.
        grouper (ExpertsGrouperForFSGPT): An object that provides group labels and usage frequencies.
        merging_layers (Optional[List[int]]): Layers to merge, if None, all layers will be merged.

    Returns:
        PhiMoEForCausalLM: The updated model with merged experts.
    """

    # 使用频率信息来分配组
    group =  [[0, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 3, 1, 2, 2, 1], [0, 1, 1, 0, 1, 0, 0, 1, 2, 0, 1, 0, 1, 3, 2, 2], [2, 0, 2, 1, 1, 0, 1, 2, 1, 0, 0, 2, 1, 0, 1, 0], [0, 1, 2, 0, 1, 0, 1, 0, 0, 2, 1, 1, 0, 2, 2, 1], [2, 1, 0, 1, 1, 1, 2, 0, 1, 0, 0, 0, 2, 2, 0, 1], [1, 1, 0, 0, 2, 3, 1, 1, 0, 1, 0, 1, 2, 0, 0, 2], [2, 0, 1, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 2, 1, 2], [1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 1, 2, 2, 1, 1], [0, 1, 0, 1, 0, 2, 2, 2, 1, 0, 3, 1, 1, 1, 0, 0], [0, 1, 1, 0, 2, 0, 1, 3, 0, 2, 1, 1, 0, 1, 0, 2], [0, 1, 2, 0, 1, 1, 2, 1, 0, 1, 0, 1, 0, 3, 0, 2], [0, 0, 1, 2, 1, 3, 2, 2, 0, 0, 1, 0, 1, 1, 1, 0], [2, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 2, 2, 2, 0, 0], [0, 0, 2, 1, 2, 1, 2, 0, 0, 1, 1, 1, 0, 2, 1, 0], [1, 1, 0, 2, 1, 0, 1, 0, 2, 2, 2, 0, 0, 1, 1, 0], [2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 1, 2, 1, 0, 1, 0], [1, 1, 3, 0, 3, 1, 0, 0, 1, 0, 2, 1, 2, 1, 0, 0], [1, 1, 3, 0, 0, 0, 2, 1, 0, 2, 0, 1, 1, 1, 0, 2], [0, 2, 2, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 2, 2], [1, 2, 3, 1, 0, 1, 3, 0, 0, 0, 0, 1, 0, 1, 2, 1], [0, 0, 1, 0, 2, 0, 1, 0, 2, 2, 1, 1, 0, 1, 1, 2], [1, 3, 0, 0, 1, 0, 1, 2, 4, 1, 2, 0, 0, 1, 1, 0], [1, 3, 0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 2, 2, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]]
    # 遍历每一层进行专家合并
    for layer_idx in range(32):
        # 获取该层的组索引和频率信息
        group_layer = group[layer_idx]  # 获取当前层的组信息
        print(layer_idx)
        # 合并当前层的专家
        model.model.layers[layer_idx].block_sparse_moe = _merge_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].block_sparse_moe,
            group=group_layer
        )
        print("done!")
    print("all done!")
    return model

# 定义钩子函数，处理每层的输出
def hook_fn(module, input, output):
    """
    钩子函数，用于捕获每层的输出并整理为长度为n_experts的一维列表。
    module: 当前层的模块
    input: 该层的输入
    output: 该层的输出
    """
    # 获取当前层的名字
    global layer_count
    layer_name = str(module)
    layer_name = f"gate_{layer_count}"  # 为了区分不同的层，我们加上一个计数器
    # 确保 router_logits 是一个标准的 Tensor 类型，并转换为 Float32
    router_logits = output.to(torch.float32).cpu().detach()  # 转换为 Float32 后提取router_logits
    
    # 计算均值之前检查是否是二维张量（即 [batch_size, num_experts] 形式）
    if len(router_logits.shape) == 2:  # 确保是二维张量
        expert_values = router_logits.mean(dim=0)  # 按专家维度进行聚合（这里求均值）
    else:
        print(f"Unexpected shape for router_logits: {router_logits.shape}")
        return
    
    # 将其转换为列表
    expert_values_list = expert_values.tolist()
    # 生成 one-hot 列表
    max_value = max(expert_values_list)  # 找到最大值
    one_hot_list = [1 if value == max_value else 0 for value in expert_values_list]

    # 检查字典中是否已经存在该层的key
    if layer_name in captured_outputs:
        # 如果已有该层的key且对应的值是list，就逐元素相加
        existing_values = captured_outputs[layer_name]
        # 逐元素相加
        for i in range(len(expert_values_list)):
            existing_values[i] += expert_values_list[i]
        captured_outputs[layer_name] = existing_values
    else:
        # 如果没有该层的key，就新建一个list
        captured_outputs[layer_name] = expert_values_list
        
    # 检查 captured_one 字典中是否已存在该层的key
    if layer_name in captured_one:
        # 如果已有该层的key且对应的值是list，就逐元素相加
        existing_one_hot = captured_one[layer_name]
        # 逐元素相加
        for i in range(len(one_hot_list)):
            existing_one_hot[i] += one_hot_list[i]
        captured_one[layer_name] = existing_one_hot
    else:
        # 如果没有该层的key，就新建一个list
        captured_one[layer_name] = one_hot_list
    # 输出捕获的router_logits的信息
    # print(f"Layer: {layer_name} - Captured router_logits: {expert_values_list}")
    # print(f"Layer: {layer_name} - One-hot representation: {one_hot_list}")
    layer_count += 1
    layer_count = layer_count % 32

results_folder = "results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)
modell = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/lm-evaluation-harness/phiMergedMoE",
    trust_remote_code=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modell = merge_by_groups_with_usage_frequency_weighting(modell)
# modell =modell.to(device)
model = HFLM(pretrained=modell,tokenizer = tokenizer, trust_remote_code=True, device="cuda")
del modell
numm = 0
for name, layer in model._model.named_modules():
    if 'gate' in name:
        layer.register_forward_hook(hook_fn)
        numm += 1
        # print(name,layer)
        # break
task_group = extract_task_groups('/root/autodl-tmp/lm-evaluation-harness/logs11.log')
task_dict = extract_task_dict('/root/autodl-tmp/lm-evaluation-harness/logs11.log', task_group)
# print(task_group)
# print(task_dict)
group_ini = 'mmlu'
task_manager = lm_eval.tasks.TaskManager()
trans = False
pree = 0.1
last_task = None
while True:
    # 每隔 5 秒执行一次（你可以调整间隔时间）
    time.sleep(5)
    pree = min(pree + random.random() * 0.2, 1)
    # 20% 的概率选择一个 group
    if random.random() < pree:
        # 从 task_group 中随机选择一个 group
        selected_group = random.choice(task_group)
        print(f"Selected group: {selected_group}")
        # captured_outputs = {}
        group_ini = selected_group
        pree = 0.1

    # 从 selected_group 中随机选择一个 task
    tasks = task_dict.get(group_ini, [])
    if tasks:
        selected_task = random.choice(tasks)
        if selected_task == last_task:
            continue
        print(f"Selected task: {selected_task}")
        captured_outputs = {}
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=model,
            tasks=[selected_task],
            num_fewshot=0,
            task_manager=task_manager,
            batch_size = 1,
            limit = 20,
        )
        # 创建一个2D数据列表
        normalized_data = []
        
        # 对每一层的值进行标准化
        for gate in captured_outputs.values():
            normalized_data.append(normalize_data(gate))
        
        # 2. 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(np.array(normalized_data), annot=True, cmap='YlGnBu', cbar=True, xticklabels=False, yticklabels=[f'Gate {i}' for i in range(32)])
        plt.title("Heatmap of Gates Data (Normalized to [0.01, 0.99])")
        plt.xlabel("Element Index")
        plt.ylabel("Gate Index")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{selected_task}_{current_time}.png"
        file_path = os.path.join(results_folder, file_name)
        # 3. 保存热力图
        plt.savefig(file_path)
        # plt.show()
        print(captured_outputs)
        print(results['results'])
        last_task = selected_task
    else:
        # if trans:
        #     break
        # if random.random() >0.5:
        #     trans = True
        continue
    



# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.