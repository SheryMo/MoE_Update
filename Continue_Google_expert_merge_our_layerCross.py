import lm_eval
import numpy as np
import math
import argparse
from sklearn.metrics.pairwise import cosine_similarity
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, List
import random
import torch.nn.functional as F  # 引入softmax函数
from task_util import *
# 初始化参数解析器
parser = argparse.ArgumentParser(description="Dataset selection for processing")
# 添加参数
parser.add_argument('--dataset', type=str, default="winogrande", help="Specify the dataset to use")
parser.add_argument('--percent', type=float, default="0.5", help="merge parts is x of the whole one")

# 解析命令行参数
args = parser.parse_args()

# 初始化一个全局字典来保存捕获的router_logits
captured_outputs = {}
captured_one = {}
layer_count = 0
# 初始化全局变量来保存捕获的expert输出
captured_expert_output = {}
layer_expert_count = 0
hook_handles = []
num_expert = 32
num_layer = 12

# 24 layers , 60 experts per layer
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
    if layer_count < 6:
        layer_name = f"encoder_gate_{layer_count*2+1}"
    else:
        layer_name = f"decoder_gate_{(layer_count-6)*2+1}"
    layer_count += 1
    layer_count = layer_count % 12
       
    # layer_name = f"gate_{layer_count}"  # 为了区分不同的层，我们加上一个计数器
    # 确保 router_logits 是一个标准的 Tensor 类型，并转换为 Float32
    router_logits = output.to(torch.float32).cpu().detach()  # 转换为 Float32 后提取router_logits
    # print('gate shape is: ')
    # print(router_logits.shape)
    # 计算均值之前检查是否是二维张量（即 [batch_size, num_experts] 形式）
    if len(router_logits.squeeze(0).shape) == 2:  # 确保是二维张量
        expert_values = F.softmax(router_logits.squeeze(0), dim=1)  # 按专家维度进行聚合（这里求均值）
        expert_values = expert_values.mean(dim=0)
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
    # layer_count += 1
    # layer_count = layer_count % 12

def hook_fn_expert(module, input, output):
    """
    钩子函数，用于捕获每层的输出并计算其soft activation。
    module: 当前层的模块
    input: 该层的输入
    output: 该层的输出
    """
    # 获取当前层的名字
    global layer_expert_count
    layer_name = str(module)
    if layer_expert_count < num_expert*num_layer/2 : 
        layer_name = f"encoder_gate_{(layer_expert_count//32)*2+1}_expert_{layer_expert_count % 32}"
    else:
        layer_name = f"decoder_gate_{((layer_expert_count - num_expert*num_layer/2)//32)*2+1}_expert_{layer_expert_count % 32}"
    layer_expert_count += 1
    layer_expert_count = layer_expert_count % (num_expert*num_layer)
    # if 'encoder' in layer_name:
    #     layer_name = f"encoder_gate_{(layer_expert_count[0]//32)*2+1}_expert_{layer_expert_count[0] % 32}"
    #     layer_expert_count[0] += 1
    #     layer_expert_count[0] = layer_expert_count[0] % (num_expert*num_layer/2)
    # if 'decoder' in layer_name:
    #     layer_name = f"decoder_gate_{(layer_expert_count[1]//32)*2+1}_expert_{layer_expert_count[1] % 32}"
    #     layer_expert_count[1] += 1
    #     layer_expert_count[1] = layer_expert_count[1] % (num_expert*num_layer/2)
    # layer_name = f"gate_{layer_expert_count//32}_expert_{layer_expert_count % 32}"  # 为了区分不同的层，我们加上一个计数器% (num_expert*num_layer)
    
    # 检查字典中是否已经存在该层的key
    if layer_name in captured_expert_output:
        if not math.isnan(captured_expert_output[layer_name][0]):
            return 
        # # 如果已有该层的key，取消该hook
        # print(f"Layer: {layer_name} already captured, removing hook.")
        # # 从module中移除hook
        # hook_hand = hook_handles[layer_expert_count]
        # hook_handles[layer_expert_count] = None
        # hook_hand.remove()
        # print(hook_handles[layer_expert_count])
        # print(f'delect {layer_expert_count}th hook')
        # return
    
    # 确保 router_logits 是一个标准的 Tensor 类型，并转换为 Float32
    router_logits = output.to(torch.float32).cpu().detach()  # 转换为 Float32 后提取router_logits
    
    # 计算softmax激活，注意：softmax通常应用于最后一个维度（即专家维度）
    if len(router_logits.shape) == 2:  # 确保是二维张量，通常是 [batch_size, num_experts]
        soft_activations = F.softmax(router_logits, dim=1)  # 按专家维度应用softmax
    else:
        print(f"Unexpected shape for router_logits: {router_logits.shape}")
        return
    
    # 计算soft activations的均值
    soft_activation_values = soft_activations.mean(dim=0)  # 按专家维度计算均值
    soft_activation_values_list = soft_activation_values.tolist()  # 转换为列表
    
    # 将soft activation存入captured_outputs
    captured_expert_output[layer_name] = soft_activation_values_list
    
    # 输出捕获的soft activation的信息
    # print(f"Layer: {layer_name} - Captured soft output: {soft_activation_values_list}")
    
    # 更新层计数器
    # layer_expert_count += 1
    # layer_expert_count = layer_expert_count % (num_expert*num_layer)  # 保持层计数在[0, 23]之间

def cross_layer_expert_merge(model, frequency_list, group, layer_idx, layer_x,names):
    """
    Merge experts across multiple layers, combining their parameters based on the frequency list weights.
    The result is assigned to the first expert of layer_idx and all other experts in this range are bound to it.
    
    Args:
        model (PhiMoEForCausalLM): The model instance.
        frequency_list (List[List[float]]): List of usage frequencies for each expert in each layer.
        group (List[List[int]]): Group labels for each expert in each layer.
        layer_idx (int): Starting layer index for the merge operation.
        layer_x (int): The layer where the merge operation ends (exclusive).
    
    Returns:
        None: The model is updated in-place.
    """
    # Collect all experts' parameters across layers layer_idx to layer_x-1
    accumulated_up_proj_weight = None
    accumulated_down_proj_weight = None
    # accumulated_gate_proj_weight = None
    total_weight = 0.0

    # Accumulate weights for the cross-layer merge
    for cross_layer_idx in range(layer_idx, layer_x):
        group_layer = group[cross_layer_idx]  # Get the group for the current layer
        frequency_layer = frequency_list[cross_layer_idx]  # Get the frequency for the current layer
        name = names[cross_layer_idx].split('_')
        
        # Loop over the experts in this layer
        for expert_idx in range(len(group_layer)):
            expert_name = 'expert_'+str(expert_idx)
            if 'encoder' in name[0]:
                expert = model.encoder.block[int(name[-1])].layer[1].mlp.experts[expert_name]
            else:
                expert = model.decoder.block[int(name[-1])].layer[2].mlp.experts[expert_name]
            # model.layers[cross_layer_idx].mlp.experts[expert_idx]
            frequency = frequency_layer[expert_idx]  # Frequency of the current expert
            
            # Add weighted parameters to the accumulator
            if accumulated_up_proj_weight is None:
                accumulated_up_proj_weight = expert.wi.weight * frequency
                accumulated_down_proj_weight = expert.wo.weight * frequency
                # accumulated_gate_proj_weight = expert.gate_proj.weight * frequency
            else:
                accumulated_up_proj_weight += expert.wi.weight * frequency
                accumulated_down_proj_weight += expert.wo.weight * frequency
                # accumulated_gate_proj_weight += expert.gate_proj.weight * frequency
            
            # Accumulate total weight
            total_weight += frequency

    # Normalize by total weight
    if total_weight > 0:
        accumulated_up_proj_weight /= total_weight
        accumulated_down_proj_weight /= total_weight
        # accumulated_gate_proj_weight /= total_weight

    # Set the merged parameters to the first expert of layer_idx
    # first_expert = model.model.layers[layer_idx].mlp.experts[0]
    first_name = names[layer_idx].split('_')
    first_layer = int(name[-1])
    first_encoder = False
    with torch.no_grad():
        if 'encoder' in first_name:
            model.encoder.block[first_layer].layer[1].mlp.experts['expert_0'].wi.weight.copy_(accumulated_up_proj_weight)
            model.encoder.block[first_layer].layer[1].mlp.experts['expert_0'].wo.weight.copy_(accumulated_down_proj_weight)
            first_encoder = True
            
        else:
            model.decoder.block[first_layer].layer[2].mlp.experts['expert_0'].wi.weight.copy_(accumulated_up_proj_weight)
            model.decoder.block[first_layer].layer[2].mlp.experts['expert_0'].wo.weight.copy_(accumulated_down_proj_weight)
            first_encoder = False
        # model.model.layers[layer_idx].mlp.experts[0].up_proj.weight.copy_(accumulated_up_proj_weight)
        # model.model.layers[layer_idx].mlp.experts[0].down_proj.weight.copy_(accumulated_down_proj_weight)
        # model.model.layers[layer_idx].mlp.experts[0].gate_proj.weight.copy_(accumulated_gate_proj_weight)

    # Bind all experts in the range layer_idx to layer_x-1 to the first expert
    for cross_layer_idx in range(layer_idx, layer_x):
        name = names[cross_layer_idx].split('_')
        for expert_idx in range(len(group[cross_layer_idx])):
            expert_name = 'expert_'+str(expert_idx)
            if 'encoder' in name[0]:
                if first_encoder == True:
                    model.encoder.block[int(name[-1])].layer[1].mlp.experts[expert_name] = model.encoder.block[first_layer].layer[1].mlp.experts['expert_0']
                else:
                    model.encoder.block[int(name[-1])].layer[1].mlp.experts[expert_name] = model.decoder.block[first_layer].layer[2].mlp.experts['expert_0']
            else:
                if first_encoder == True:
                    model.decoder.block[int(name[-1])].layer[2].mlp.experts[expert_name] = model.encoder.block[first_layer].layer[1].mlp.experts['expert_0']
                else:
                    model.decoder.block[int(name[-1])].layer[2].mlp.experts[expert_name] = model.decoder.block[first_layer].layer[2].mlp.experts['expert_0']
            
            # model.model.layers[cross_layer_idx].mlp.experts[expert_idx] = model.model.layers[layer_idx].mlp.experts[0]
    
    print(f"Cross-layer merge completed for layers {layer_idx} to {layer_x-1}")
    return model
    
def _merge_experts_by_usage_frequency_weighting(
        ffn,
        group,  
        frequency_list  
):
    """
    Merge experts within a group by usage frequency weighting.
    All experts in a group will share the same parameters after merging.

    Args:
        ffn (PhiMoESparseMoeBlock): The PhiMoE block containing experts.
        group (List[int]): Group labels indicating which group each expert belongs to (length = 16).
        frequency_list (List[float]): Usage frequencies for each expert (length = 16).

    Returns:
        PhiMoESparseMoeBlock: The updated PhiMoE block with merged experts.
    """
    assert len(group) == len(frequency_list) == len(ffn.experts)

    # Convert group and frequency list to tensors for easier processing
    group_tensor = torch.tensor(group, dtype=torch.long)
    frequency_tensor = torch.tensor(frequency_list, dtype=torch.float)

    for label in group_tensor.unique():
        # Find the indices of experts in this group
        expert_indices = torch.where(group_tensor == label)[0]
        print(expert_indices)
        expert_indices_int = expert_indices.tolist()
        with torch.no_grad():
            expert_names = ['expert_'+str(expert_idx) for expert_idx in expert_indices_int]
            # Accumulate weighted parameters for the group
            w1_weight_list = torch.stack(
                [ffn.experts[expert_names[i]].wi.weight * frequency_tensor[expert_indices[i]] for i in range(len(expert_indices_int))], dim=0
            )
            w2_weight_list = torch.stack(
                [ffn.experts[expert_names[i]].wo.weight * frequency_tensor[expert_indices[i]] for i in range(len(expert_indices_int))], dim=0
            )

            # Normalize the weights by their sum
            total_weight = torch.sum(frequency_tensor[expert_indices])
            w1_weight = torch.sum(w1_weight_list, dim=0) / (total_weight )
            w2_weight = torch.sum(w2_weight_list, dim=0) / (total_weight )
            
            # Set the merged weight to the first expert in the group
            ffn.experts['expert_'+str(int(expert_indices[0]))].wi.weight.copy_(w1_weight)
            ffn.experts['expert_'+str(int(expert_indices[0]))].wo.weight.copy_(w2_weight)

            # Bind all experts in the group to the first expert (sharing parameters)
            for expert_idx in expert_names[1:]:
                ffn.experts[expert_idx] = ffn.experts[expert_names[0]]
            print(expert_indices[0])
    
    return ffn

def adjust_groups_based_on_variance_similarity(frequency_list, group):
    """
    Adjust the groupings across layers based on frequency variance similarity between layers.
    
    Args:
        frequency_list (List[List[float]]): 32x16 list of frequencies for each expert in each layer.
        group (List[List[int]]): Group labels for each expert in each layer.
    
    Returns:
        List[List[int]]: Updated group labels after processing the variance similarity.
    """
    num_layers = len(frequency_list)
    notdone = True
    # Step 1: Compute the variance for each layer based on its frequency list
    layer_variance_list = []
    for layer_frequencies in frequency_list:
        layer_variance = compute_frequency_variance(layer_frequencies)
        layer_variance_list.append(layer_variance)
    base_sim = 0.8
    while notdone: 
        # Step 2: Compare layers for variance similarity starting from the last layer
        for layer_idx in range(num_layers - 1, 2, -1):
            # Calculate the number of groups across layers
            total_groups = 0
            for layer_index in range(num_layers):
                current_layer_groups = set(group[layer_index]) - {-2}
                if -2 in group[layer_index]:
                    subsequent_layers = layer_index
                    while subsequent_layers < num_layers and -2 in group[subsequent_layers]:
                        subsequent_layers += 1
                    
                    # Count the layers with -2 as a single group
                    total_groups += 1
                    layer_index = subsequent_layers - 1
                else:
                    total_groups += len(current_layer_groups)
            if total_groups <= (num_expert * num_layer *args.percent):
                print(f"Total groups {total_groups} exceeded the threshold, stopping comparison.")
                notdone = False
                break
            
            current_variance = layer_variance_list[layer_idx]
            previous_variance = layer_variance_list[layer_idx - 1]
            
            # Step 3: Compute the similarity between the current layer and the previous layer
            similarity = compute_similarity_between_layers(current_variance, previous_variance)
            print(similarity)
            # Step 4: If similarity exceeds the threshold (0.8), set both groups to -2
            if similarity > base_sim:
                # Set the groups of both layers to -2
                for expert_idx in range(len(group[layer_idx])):
                    group[layer_idx][expert_idx] = -2
                for expert_idx in range(len(group[layer_idx - 1])):
                    group[layer_idx - 1][expert_idx] = -2
        base_sim = base_sim/2
    return group
    
def assign_experts_to_groups_by_similarity(
        frequency_list,
    expert_output, 
) :
    """
    根据专家的原始频率信息，选择每层中最大频率的专家作为定点，
    并将其他专家与定点专家进行相似度比较，按照相似度超过0.7来分组。

    Args:
        frequency_list (list of list of float): 32x16 的列表，表示32层中每层16个专家的频率。

    Returns:
        group (list of list of int): 32x16 的列表，表示每层16个专家所属的组索引。
    """
    num_layers = len(frequency_list)
    num_experts = len(frequency_list[0])  # 每层有60个专家
    print(num_experts)
    group = [[-1 for _ in range(num_experts)] for _ in range(num_layers)]  # 初始化组标签
    limit = num_experts  # 组的最大数量
    # 遍历每一层
    for layer_idx in range(num_layers):
        if layer_idx == int(num_layers/4):
            limit = limit/4*3
        if layer_idx == int(num_layers/2):
            limit = limit/3*2
        if layer_idx == int(num_layers/4*3):
            limit = limit/2
        layer_frequencies = frequency_list[layer_idx]
        layer_expert_output = expert_output[layer_idx*num_experts:(layer_idx+1)*num_experts]
        # 初始化组
        current_group_idx = -1
        # 记录当前层的所有分组情况，找出未分组的专家
        ungrouped_experts = list(range(num_experts))
        while current_group_idx < limit:
            current_group_idx += 1
            if current_group_idx == limit:
                break
            anchor_expert_idx = max(ungrouped_experts, key=lambda idx: layer_frequencies[idx])
            group[layer_idx][anchor_expert_idx] = current_group_idx
            ungrouped_experts.remove(anchor_expert_idx)
        current_group_idx = -1
        while True:
            ungrouped_experts = [idx for idx, value in enumerate(group[layer_idx]) if value == -1]
            if not ungrouped_experts:  # 如果没有未分组的专家，停止分组
                break
            current_group_idx +=1
            if current_group_idx >= limit:
                break
            # 选择当前未分组的最大频率专家作为定点
            anchor_expert_idx = [idx for idx, value in enumerate(group[layer_idx]) if value == current_group_idx]
            anchor_expert_idx = anchor_expert_idx[0]
            anchor_expert_output = layer_expert_output[anchor_expert_idx]
            # group[layer_idx][anchor_expert_idx] = current_group_idx  # 将定点专家归为第一个组
            numm = 0
            cos_simi = [-1 for _ in range(num_experts)]
            # 遍历其他未分组专家，与定点专家计算相似度
            for expert_idx in ungrouped_experts:
                current_expert_output = layer_expert_output[expert_idx]
                
                # 计算当前专家和定点专家之间的余弦相似度
                cos_sim = cosine_similarity(
                    np.array(anchor_expert_output).reshape(1, -1),
                    np.array(current_expert_output).reshape(1, -1)
                )[0][0]
                
                cos_simi[expert_idx] = cos_sim
            # 将相似度超过0.7的专家分到当前组
            while True:
                best_match_idx = np.argmax(cos_simi)
                if cos_simi[best_match_idx] == -1:
                    break
                if layer_expert_output[best_match_idx][0] == 0:
                    cos_simi[best_match_idx] = -1
                    continue
                group[layer_idx][best_match_idx] = current_group_idx
                ungrouped_experts.remove(best_match_idx)  # 将已分组专家从未分组列表中移除
                break
        ungrouped_experts = [idx for idx, value in enumerate(group[layer_idx]) if value == -1]
        # 如果有专家未分组，则将其归为最后一组
        if ungrouped_experts:
            for expert_idx in ungrouped_experts:
                group[layer_idx][expert_idx] = limit - 1  # 将剩余专家归为最后一组
            ungrouped_experts.clear()  # 清空未分组专家列表
                    
    return group
    
def merge_by_groups_with_usage_frequency_weighting(
        model,
        frequency_list,
    expert_output, 
    names
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
    group = assign_experts_to_groups_by_similarity(frequency_list, expert_output)
    group = adjust_groups_based_on_variance_similarity(frequency_list, group)
    
    print(group)
    print("1")
    num_layers = len(frequency_list)
    layer_idx = 1
    while layer_idx < num_layers:
        # Check if the current layer contains any -2 in its group
        if -2 in group[layer_idx]:
            # If -2 is found, look for the next layer where -2 is not present
            layer_x = layer_idx
            while layer_x < num_layers and -2 in group[layer_x]:
                layer_x += 1
            
            # If we found a valid layer_x, perform cross-layer merge from layer_idx to layer_x-1
            if layer_x <= num_layers:
                model = cross_layer_expert_merge(model, frequency_list, group, layer_idx, layer_x,names)
                # After merging, update layer_idx to layer_x (next unmerged layer)
                layer_idx = layer_x
        else:
            # If no -2 in the group, proceed with the original logic
            name_layer = names[layer_idx].split('_')
            group_layer = group[layer_idx]  # Get the group for the current layer
            frequency_layer = frequency_list[layer_idx]  # Get frequency for the current layer
            print(f"Normal merging for layer {layer_idx}")
            if 'encoder' in name_layer[0]:
                model.encoder.block[int(name_layer[-1])].layer[1].mlp = _merge_experts_by_usage_frequency_weighting(
                ffn=model.encoder.block[int(name_layer[-1])].layer[1].mlp,
                group=group_layer,
                frequency_list=frequency_layer,
            )
            else:
                model.decoder.block[int(name_layer[-1])].layer[2].mlp = _merge_experts_by_usage_frequency_weighting(
                ffn=model.decoder.block[int(name_layer[-1])].layer[2].mlp,
                group=group_layer,
                frequency_list=frequency_layer,
            )
            # # Merge experts for the current layer
            # model.model.layers[layer_idx].mlp = _merge_experts_by_usage_frequency_weighting(
            #     ffn=model.model.layers[layer_idx].mlp,
            #     group=group_layer,
            #     frequency_list=frequency_layer,
            # )
            layer_idx += 1  # Move to the next layer
        print("done!")
    print("all done!")
    return model
    
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
    
frequence_dict = {}

task_group = extract_task_groups('/root/autodl-tmp/lm-evaluation-harness/logs11.log')
task_dict = extract_task_dict('/root/autodl-tmp/lm-evaluation-harness/logs11.log', task_group)

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32", trust_remote_code=True)
task_manager = lm_eval.tasks.TaskManager()

# # Setting `task_manager` to the one above is optional and should generally be done
# # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
# results = lm_eval.simple_evaluate( # call simple_evaluate
#     model=model,
#     tasks=["mmlu_formal_logic"],
#     num_fewshot=0,
#     task_manager=task_manager,
#     batch_size = 1,
#     limit = 50,
    
# )

model = HFLM(pretrained="/root/autodl-tmp/lm-evaluation-harness/google/switch-base-32", trust_remote_code=True,device = 'cuda')
numm = 0
# print(model._model)
for name, layer in model._model.named_modules():
    if 'mlp.router.classifier' in name:
        print(name)
        layer.register_forward_hook(hook_fn)
        numm += 1
        # print(name,layer)
        # break
    if '.wi' in name and 'expert_' in name:
        handle = layer.register_forward_hook(hook_fn_expert)
        hook_handles.append(handle)
        
print(f'all_gate number:{numm}')
results = lm_eval.simple_evaluate( # call simple_evaluate squad_completion
    model=model,
    tasks=[args.dataset],
    num_fewshot=0,
    task_manager=task_manager,
    batch_size = 1,
    limit = 100,
)
print('full model:')
print(results['results'])
print(captured_outputs)
# print(captured_expert_output)
frequency_list,names_fre = map_to_range(captured_outputs,[])
expert_output,names_out = map_to_range(captured_expert_output,[])
print(names_fre)
modell = merge_by_groups_with_usage_frequency_weighting(model._model, frequency_list, expert_output,names_fre)
modell = modell.cuda()
model = HFLM(pretrained=modell, trust_remote_code=True, device="cuda")
model_size = get_model_size(modell)
print(f"Model size: {model_size:.4f} GB")
# print("begin to save!")
# # modell.save_pretrained('/root/autodl-tmp/lm-evaluation-harness/HiphiMergedMoE')

# model = HFLM(pretrained=modell, trust_remote_code=True, device='cpu')


# # Setting `task_manager` to the one above is optional and should generally be done
# # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
results = lm_eval.simple_evaluate( # call simple_evaluate squad_completion
    model=model,
    tasks=[args.dataset],
    num_fewshot=0,
    task_manager=task_manager,
    batch_size = 1,
    limit = 10000,
)
# print(captured_outputs)
# print(captured_one)
print(results['results'])