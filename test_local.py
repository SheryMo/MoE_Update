 # python>=3.10

import torch
import torch.nn as nn
import os
import random
import numpy as np
from torch.utils.data import Dataset
from typing import List
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict
from scipy.optimize import linprog
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    T5TokenizerFast,
    get_scheduler,
    SwitchTransformersForConditionalGeneration as HFSwitch,
)
from transformers.utils import logging as hf_logging
from collections import defaultdict
from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForSeq2Seq,
    get_evaluate_fn,
    EXTRA_KEYS_FOR_EVAL,
    keep_only_supporting_facts_in_context_for_hotpotqa
)
from mcsmoe.merging import (
    ExpertsGrouperForSwitch,
    merge_by_groups_with_usage_frequency_weighting
)
from mcsmoe.models import (
    SwitchTransformersWrapperForDistillation
)
from mcsmoe.utils.sparsity import compute_weight_stable_rank
from mcsmoe.utils.training_utils import freeze_switch_routers_for_finetuning

accelerator = Accelerator(gradient_accumulation_steps=1)

def get_expert_usage_frequency_and_evaluation(task, model, tokenizer, preprocessing_num_workers=4, num_samples_for_merging=512, per_device_train_batch_size=8, per_device_eval_batch_size=8, similarity_base='cosine', EXTRA_KEYS_FOR_EVAL=EXTRA_KEYS_FOR_EVAL, num_eval_steps=1000):
    """
    获取模型在指定任务上的 Expert 使用频率并返回模型评估结果。

    :param task: 任务名称（例如 'sst2', 'mnli', 等）
    :param model: 训练好的模型
    :param tokenizer: 用于数据预处理的tokenizer
    :param preprocessing_num_workers: 数据预处理的并行工作数
    :param num_samples_for_merging: 用于合并的样本数量
    :param per_device_train_batch_size: 训练集的批次大小
    :param per_device_eval_batch_size: 验证集的批次大小
    :param similarity_base: 相似度计算的基准（例如 'cosine'）
    :param EXTRA_KEYS_FOR_EVAL: 额外的评估键（如果有）
    :param num_eval_steps: 每次评估的步数
    :return: expert 使用频率的字典和评估结果
    """
    # 加载数据集
    
    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task],trust_remote_code=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=tokenizer.model_max_length,
                                           return_tensors='pt',
                                           keys_to_ignore=EXTRA_KEYS_FOR_EVAL)
    # 获取训练集和验证集
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"] if task != "mnli" else (
        raw_dataset["validation_matched"], raw_dataset["validation_mismatched"]
    )

    # 数据预处理和token化
    with accelerator.main_process_first():
        if task == "hotpotqa":
            train_dataset = train_dataset.map(
                keep_only_supporting_facts_in_context_for_hotpotqa,
                batched=False,
                num_proc=preprocessing_num_workers
            )
            eval_dataset = eval_dataset.map(
                keep_only_supporting_facts_in_context_for_hotpotqa,
                batched=False,
                num_proc=preprocessing_num_workers
            )

        train_dataset = train_dataset.map(
            Seq2SeqDataPreProcessor(benchmark=task),
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=eval_dataset.column_names
        )

    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=False),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )
    dataset_size = len(tokenized_train_dataset)
    num_samples_for_merging = min(num_samples_for_merging, dataset_size)
    # 随机选择用于合并的数据集子集
    dataset_for_merging = tokenized_train_dataset.shuffle(seed=2333).select(range(num_samples_for_merging))

    # 创建 DataLoader
    dataloader_for_merging = DataLoader(
        dataset_for_merging,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=num_samples_for_merging,
        num_workers=3
    )

    # 假设 grouper 是一个用于计算使用频率的工具
    grouper = ExpertsGrouperForSwitch(
        config=model.config,
        similarity_base='router-logits'
    )

    # 计算模型的相似度和专家使用频率
    grouper.compute_all_similarities(
        model=model,
        batch=next(iter(dataloader_for_merging)),
    )
    grouper.compute_all_usages(
        model=model,
        batch=next(iter(dataloader_for_merging)),
    )
    core_experts = grouper.group_experts_into_clusters_by_routing_guided_globally(
        average_num_groups=4,
        merging_encoder_layers=[1,3,5,7,9,11],
        merging_decoder_layers=[1,3,5,7,9,11]
    )
    
    # 获取并返回 usage_frequency_dict
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    group_dict = grouper.group_state_dict()

    # ========================= Evaluate ================================
    # 获取评估函数
    evaluate_fn = get_evaluate_fn(
        task=task,
        tokenizer=tokenizer,
        raw_eval_dataset=raw_dataset["validation"]
    )

    # 创建验证集 DataLoader
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=per_device_eval_batch_size,
        num_workers=3
    )


    model.eval()
    losses = []
    output_labels = []
    output_predictions = []
    output_ids = [] if task in ["squad", "copa", "multirc", "squad_v2", "hotpotqa"] else None
    for eval_step, eval_batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=num_eval_steps):
        extra_keys_eval_batch = {}
        for key in list(eval_batch.keys()):
            if key in EXTRA_KEYS_FOR_EVAL:
                extra_keys_eval_batch[key] = eval_batch.pop(key)
        eval_batch = {key: value.to(accelerator.device) if isinstance(value, torch.Tensor) else value 
                      for key, value in eval_batch.items()}
        with torch.no_grad():
            outputs = model(**eval_batch) # 将 eval_batch 移到正确的设备
        
        eval_labels = accelerator.gather(eval_batch['labels'])
        output_labels += torch.cat([ 
            eval_labels,
            torch.ones(eval_labels.shape[0], tokenizer.model_max_length - eval_labels.shape[1],
                       dtype=eval_labels.dtype,
                       device=eval_labels.device) * -100
        ], dim=-1)
        
        eval_logits = accelerator.gather(outputs.logits)
        output_predictions += eval_logits.argmax(dim=-1).tolist()
        
        if task in ["squad", "squad_v2", "hotpotqa"]:
            output_ids += extra_keys_eval_batch["id"]
        elif task == "copa" or task == "multirc":
            output_ids += extra_keys_eval_batch["idx"]
        # print(outputs.keys())
        losses.append(accelerator.gather_for_metrics(outputs["loss"]).unsqueeze(0))
        
        if eval_step >= num_eval_steps:
            break

    losses = torch.cat(losses, dim=0)
    eval_loss = torch.mean(losses)
    output_labels = torch.stack(output_labels, dim=0)

    # 获取评估结果
    eval_res = evaluate_fn(predictions=output_predictions, labels=output_labels, ids=output_ids)
    eval_res["task_loss"] = eval_loss.item()
    print(eval_res)
    
    return usage_frequency_dict, eval_res, group_dict

def get_updated_experts(usage_frequency_dict, group_dict):
    result_dict = {}
    
    # 遍历每一层
    for layer_key, group_tensor in group_dict.items():
        # 获取当前层的频率信息
        frequencies = usage_frequency_dict[layer_key]
        
        # 获取当前层组别的唯一值
        unique_groups = group_tensor.unique().tolist()
        
        # 创建一个字典，用于存储每个组的结果
        layer_result = {}
        
        # 遍历所有组别
        for group in unique_groups:
            # 获取该组对应的expert索引
            group_indices = (group_tensor == group).nonzero(as_tuple=False).squeeze().tolist()
            
            # 计算该组内的频率权重
            group_weights = [frequencies[idx].item() for idx in group_indices]
            
            # 计算该组的带权加和
            weighted_sum = sum(group_weights)
            
            # 归一化权重（除以带权加和）
            if weighted_sum > 0:  # 防止除零错误
                normalized_weights = [weight / weighted_sum for weight in group_weights]
            else:
                normalized_weights = group_weights  # 如果加和为零，保持原样（或可以处理为0）
            
            # 将每一组的结果存入结果字典
            layer_result[group] = {
                'indices': group_indices,
                'weights': normalized_weights,
                'weighted_sum': weighted_sum
            }
        
        # 将该层的结果存入最终结果字典
        result_dict[layer_key] = layer_result
    
    return result_dict

def find_experts_in_directory(directory_path = "/root/autodl-fs/MC-SMoE_Follow/experts_checkpoints/"):
    """
    Finds all expert files in the specified directory, and returns them categorized by layers (encoder/decoder) 
    and expert index in the required format where each expert is treated as a weight of 1.

    Args:
    - directory_path (str): Path to the directory containing the expert files.

    Returns:
    - dict: A dictionary where keys are layers (e.g., 'encoder.block.1.layer.1.mlp') and values are dictionaries
            with expert indices as keys and a dictionary of indices, weights, and weighted_sum as values.
    """
    expert_dict = defaultdict(dict)  # A dictionary to hold the expert info categorized by layers
    
    # Define the pattern to match the filenames, assuming the pattern 'encoder_blockX_layerY_expert_Z.pt'
    # and 'decoder_blockX_layerY_expert_Z.pt'
    pattern = re.compile(r"^(encoder|decoder)_block(\d+)_layer(\d+)_expert_(\d+)\.pt$")
    
    # Walk through the directory and check each file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            match = pattern.match(file)
            if match:
                # Extract type (encoder/decoder), block, layer, and expert index from the matched filename
                model_type = match.group(1)  # 'encoder' or 'decoder'
                block = int(match.group(2))  # block number
                layer = int(match.group(3))  # layer number
                expert_index = int(match.group(4))  # expert index
                
                # Prepare the layer name, use '.' to separate block and layer
                layer_name = f"{model_type}.block{block}.layer{layer}.mlp"
                
                # Prepare the structure for the expert under that layer
                expert_dict[layer_name][expert_index] = {
                    'indices': [expert_index],
                    'weights': [1],  # Each expert has a weight of 1
                    'weighted_sum': 1.0  # The sum of weights is 1.0 for each individual expert
                }
    
    # Convert defaultdict to a regular dict for a cleaner output
    return dict(expert_dict)

def can_form_using_linear_combination(dict1, dict2, num_experts=32):
    """
    Check if each group in dict2 can be represented as a linear combination of the groups in dict1.
    If so, return the coefficients of the linear combination, and check if the entire model has a solution.
    
    Args:
    - dict1 (dict): The existing expert groups.
    - dict2 (dict): The target expert groups.
    - num_experts (int): The total number of experts (dimensions of the vector space).
    
    Returns:
    - dict: A dictionary where each layer contains a list of dictionaries. 
            Each dictionary contains a boolean (whether a solution exists) and if true, 
            the linear combination coefficients of the expert groups in dict1.
    - bool: Whether the entire model has a solution (True or False).
    """
    results = {}
    model_has_solution = True  # Assume the model has a solution, will update based on the layers

    # Iterate through each layer in dict2
    for layer, target_groups in dict2.items():
        layer_results = []

        # For each target group in dict2, try to find a linear combination from dict1
        for target_idx, target_group in target_groups.items():
            # Convert the target group to a sparse vector representation
            target_vector = np.zeros(num_experts)
            for idx, weight in zip(target_group['indices'], target_group['weights']):
                target_vector[idx] = weight

            # Create the matrix for the linear system, where each column is a group in dict1
            dict1_vectors = []
            for idx, group in dict1.get(layer, {}).items():
                # Convert each dict1 group to a sparse vector representation
                dict1_vector = np.zeros(num_experts)
                for idx_, weight in zip(group['indices'], group['weights']):
                    dict1_vector[idx_] = weight
                dict1_vectors.append(dict1_vector)

            # If dict1 contains no groups for this layer, we can't form a combination
            if len(dict1_vectors) == 0:
                layer_results.append({'solution_exists': False, 'coefficients': None})
                model_has_solution = False  # No solution for this layer means the model doesn't have a solution
                continue

            # Convert dict1_vectors to a matrix (each column is a vector)
            A = np.array(dict1_vectors).T  # Matrix of expert vectors from dict1
            b = target_vector  # The target vector from dict2

            # Solve the linear system A * w = b (least squares if no exact solution)
            try:
                weights, residuals, rank, s = lstsq(A, b)
                # If the residual is close to zero, then the solution exists
                if np.allclose(np.dot(A, weights), b):
                    layer_results.append({
                        'solution_exists': True,
                        'coefficients': weights.tolist()  # Coefficients of the linear combination
                    })
                else:
                    layer_results.append({
                        'solution_exists': False,
                        'coefficients': None
                    })
                    model_has_solution = False  # No solution for this group, model doesn't have a solution
            except Exception as e:
                layer_results.append({
                    'solution_exists': False,
                    'coefficients': None
                })
                model_has_solution = False  # No solution for this layer means the model doesn't have a solution

        # Store the result for this layer
        results[layer] = layer_results

    return results, model_has_solution

def merge_experts(dict1, dict2):
    """
    Merge two dictionaries containing expert information by combining the groups for each layer.
    If a layer exists in both dictionaries, reindex the combined groups.
    
    Args:
    - dict1 (dict): The first dictionary containing experts.
    - dict2 (dict): The second dictionary containing experts.
    
    Returns:
    - dict: A dictionary where each layer contains merged expert groups with reindexed group numbers.
    """
    merged_dict = defaultdict(dict)  # To store the merged expert information
    
    # Combine the layers from both dictionaries
    all_layers = set(dict1.keys()).union(set(dict2.keys()))  # All unique layer names
    
    for layer in all_layers:
        group_dict1 = dict1.get(layer, {})  # Get the groups for the layer from dict1, or empty if not found
        group_dict2 = dict2.get(layer, {})  # Get the groups for the layer from dict2, or empty if not found
        
        # Merging groups from both dictionaries for this layer
        merged_groups = {}
        
        # Add groups from dict1
        for idx, group in group_dict1.items():
            merged_groups[idx] = {
                'indices': group['indices'],
                'weights': group['weights'],
                'weighted_sum': group['weighted_sum']
            }
        
        # Add groups from dict2
        for idx, group in group_dict2.items():
            # If the same group index exists, merge indices and weights
            new_idx = len(merged_groups)  # New index for this group in merged dict
            merged_groups[new_idx] = {
                'indices': group['indices'],
                'weights': group['weights'],
                'weighted_sum': group['weighted_sum']
            }
        
        # Calculate weighted_sum for each group in the merged layer
        for idx, group in merged_groups.items():
            group['weighted_sum'] = sum(group['weights'])  # Sum of weights for the group
        
        # Add the merged groups to the merged dictionary for this layer
        merged_dict[layer] = merged_groups
    
    return dict(merged_dict)

def find_remote_experts():
    ### TODO
    return None

def find_missing_experts(now_expert, updated_expert):
    existing_experts = find_experts_in_directory()
    Linear_break = True
    remote = False
    remote_num = 0
    results = None
    remote_experts = {}
    while(Linear_break):
        if remote:
            print("Remote operation required. Triggering remote update.")
            existing_experts = find_experts_remote() ##TODO: find remote experts
            if remote_experts == {}:
                remote_experts = existing_experts
            else:
                remote_experts = merge_experts(remote_experts, existing_experts)
            remote_num = remote_num+1
        now_expert = merge_experts(existing_experts, now_expert)
        results, model_has_solution = can_form_using_linear_combination(now_expert, updated_expert)
        if model_has_solution:
            Linear_break = False
        remote = True
    return results, now_expert, remote_experts

# 从文件加载专家到模型
def load_expert_from_file(file_path, model, block_index, layer_index, expert_index):
    """
    从指定路径加载专家并更新到模型中
    :param file_path: 专家文件的路径
    :param model: 模型对象
    :param block_index: 块的索引
    :param layer_index: 层的索引
    :param expert_index: 专家的索引
    """
    save_dir = 'experts_checkpoints'
    file_path = os.path.join(save_dir, f'decoder_block{block_index}_layer2_{expert_name}.pt')
    expert_state_dict = torch.load(file_path)
    model.encoder.block[block_index].layer[layer_index].mlp.experts[expert_index].load_state_dict(expert_state_dict)
    print(f"Expert {expert_index} loaded from {file_path}")

# 将显存中的专家保存回文件
def save_expert_to_file(expert, file_path):
    """
    保存显存中的专家到指定文件
    :param expert: 需要保存的专家模型
    :param file_path: 保存文件的路径
    """
    torch.save(expert.state_dict(), file_path)
    print(f"Expert saved to {file_path}")

# 本地更新操作：从主存加载专家到显存
def local_doing(model, block_index, layer_index, expert_index, storage_dir):
    """
    本地操作：将指定专家从主存加载到显存并保存到文件
    :param model: 模型对象
    :param block_index: 块的索引
    :param layer_index: 层的索引
    :param expert_index: 专家的索引
    :param storage_dir: 存储目录
    """
    # 获取模型中的专家
    expert_model = model.encoder.block[block_index].layer[layer_index].mlp.experts[expert_index]  # 获取专家模型
    
    # 模拟保存该专家到指定目录
    expert_path = os.path.join(storage_dir, f"expert_{block_index}_{layer_index}_{expert_index}_test.pt")
    torch.save(expert_model.state_dict(), expert_path)  # 保存该专家的权重
    print(f"Expert {block_index}-{layer_index}-{expert_index} saved at {expert_path}.")

# 定期推理并进行 local/remote 操作
def perform_local_remote_operations(model, tokenizer, model_pattern, group_pattern, storage_dir='experts_checkpoints', threshold=0.9, task_keys=None, eval_steps=1000):
    """
    定期从 task_keys 中随机选择一个任务进行推理，比较专家的使用频率模式和模型的模式，进行 local 或 remote 操作。
    """
    task = random.choice(task_keys)  # 从任务列表中随机选择一个任务

    print(f"Performing inference on task: {task}")

    # 获取专家使用频率模式和评估结果
    expert_frequency_pattern, eval_res, group_dict = get_expert_usage_frequency_and_evaluation(
        task=task,
        model=model,
        tokenizer=tokenizer,
        num_eval_steps=eval_steps
    )
    print(group_dict)
    # 加载模型的模式
    # model_pattern = load_model_pattern(file_path=os.path.join(storage_dir, "model_pattern.pt"))
    
    # 计算模式差异
    change = compute_pattern_difference(model_pattern, expert_frequency_pattern)

    print(f"Pattern difference: {change}")

    if change > threshold:
        now_expert = get_updated_experts(model_pattern,group_pattern)
        updated_expert = get_updated_experts(expert_frequency_pattern,group_dict)
        results, need_expert,remote_experts = find_missing_experts(now_expert, updated_expert)
        if remote_experts == {}:
            print("Local operation: Updating expert weights.")
        else:
            print("Remote operation required. Triggering remote update.")

        # 更新本地pattern -- 最后操作（更新完成后）
        model_pattern = expert_frequency_pattern
        group_pattern = group_dict
    #更新完成后还需要在本地主存判断是否需要更新存储的东西 -- 本地存储默认存储原来的experts，如果需要存储merged experts，相关函数需要更新（本地文件拿取experts pattern的函数+local doing？）
    # 返回评估结果
    return eval_res, model_pattern, group_pattern

def generate_file_path(task, base_dir='/root/autodl-fs/MC-SMoE_Follow/results'):
    """
    生成文件路径，task 为任务名称，根据任务决定路径中 xxx 的部分。
    :param task: 任务名称
    :param base_dir: 基础目录，默认设置为 autdlf-fs/MC-SMoE_Follow/results
    :return: 生成的文件路径
    """
    if task == 'copa': 
        task_path = 'copa/merged4_1'
    else:
        task_path = f'{task}/merged4'

    # 组合成完整路径
    file_path = os.path.join(base_dir, task_path, 'normal/router-logits/frequence_state_dict.pt')
    return file_path

def generate_file_path_1(task, base_dir='/root/autodl-fs/MC-SMoE_Follow/results'):
    """
    生成文件路径，task 为任务名称，根据任务决定路径中 xxx 的部分。
    :param task: 任务名称
    :param base_dir: 基础目录，默认设置为 autdlf-fs/MC-SMoE_Follow/results
    :return: 生成的文件路径
    """
    if task == 'copa': 
        task_path = 'copa/merged4_1'
    else:
        task_path = f'{task}/merged4'

    # 组合成完整路径
    file_path = os.path.join(base_dir, task_path, 'normal/router-logits/group_state_dict.pt')
    return file_path

# 模拟从文件中加载模型模式
def load_model_pattern(file_path="model_pattern.pt"):
    """
    加载模型的模式（存储在文件中）。
    :param file_path: 模式文件路径
    :return: 模型的模式表示
    """
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        print(f"Model pattern file not found at {file_path}, creating a dummy pattern.")
        return {f'encoder_layer_{i}': torch.randn(1) for i in range(12)}



def compute_pattern_difference(model_pattern1, model_pattern2):
    distances = []
    
    # 遍历所有的层
    for layer_name in model_pattern1:
        if layer_name in model_pattern2:
            tensor1 = model_pattern1[layer_name]
            tensor2 = model_pattern2[layer_name]
            
            # 计算该层的欧几里得距离
            distance = torch.norm(tensor1 - tensor2).item()
            distances.append(distance)
    
    # 计算所有层的平均距离
    avg_distance = np.mean(distances)
    print(avg_distance)
    return avg_distance



# 主程序：定期执行推理并进行模式比较
def main():
    # 初始化任务键和其他参数
    task_keys = [
        "cola", "stsb", "rte", "sst2", "qqp", "qnli", "mrpc", 
        "multirc", "squad", "copa", "winogrande", 
        "wikiqa", "hotpotqa"
    ]#, "triviaqa", "mnli", "openbookqa", "squad_v2", "hellaswag"
    tokenizer = T5TokenizerFast.from_pretrained("google/switch-base-32")
    checkpoint = "results/copa/merged4_1/normal/router-logits/latest"  # 模型检查点路径
    storage_dir = '/root/autodl-fs/MC-SMoE_Follow/experts_checkpoints'  # 存储专家的路径
    os.makedirs(storage_dir, exist_ok=True)  # 创建存储目录
    initial_task = 'copa'

    # 加载模型
    model = HFSwitch.from_pretrained(checkpoint)
    model_pattern  = load_model_pattern(file_path=generate_file_path(initial_task))
    group_pattern= load_model_pattern(file_path=generate_file_path_1(initial_task))
    # 定期进行推理
    for epoch in range(10):  # 假设进行10个epoch
        print(f"Epoch {epoch + 1}")
        eval_res, model_pattern, group_pattern = perform_local_remote_operations(
            model=model,
            tokenizer = tokenizer,
            model_pattern = model_pattern,
            group_pattern = group_pattern,
            storage_dir=storage_dir,
            task_keys=task_keys,
            eval_steps=1000,  # 每个任务进行1000步评估

        )
        print(f"Evaluation results: {eval_res}")

if __name__ == "__main__":
    main()

