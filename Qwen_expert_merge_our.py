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
    layer_name = f"gate_{layer_count}"  # 为了区分不同的层，我们加上一个计数器
    # 确保 router_logits 是一个标准的 Tensor 类型，并转换为 Float32
    router_logits = output.to(torch.float32).cpu().detach()  # 转换为 Float32 后提取router_logits
    
    # 计算均值之前检查是否是二维张量（即 [batch_size, num_experts] 形式）
    if len(router_logits.shape) == 2:  # 确保是二维张量
        expert_values = F.softmax(router_logits, dim=1)  # 按专家维度进行聚合（这里求均值）
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
    layer_count += 1
    layer_count = layer_count % 24

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
    layer_name = f"gate_{layer_expert_count//60}_expert_{layer_expert_count % 60}"  # 为了区分不同的层，我们加上一个计数器
    
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
    layer_expert_count += 1
    layer_expert_count = layer_expert_count % (24*60)  # 保持层计数在[0, 23]之间

def _merge_experts_by_usage_frequency_weighting(
        ffn,
        group,  # Group indices, length = 16
        frequency_list  # Usage frequencies, length = 16
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
        
        with torch.no_grad():
            # Accumulate weighted parameters for the group
            w1_weight_list = torch.stack(
                [ffn.experts[expert_idx].up_proj.weight * frequency_tensor[expert_idx] for expert_idx in expert_indices], dim=0
            )
            w2_weight_list = torch.stack(
                [ffn.experts[expert_idx].down_proj.weight * frequency_tensor[expert_idx] for expert_idx in expert_indices], dim=0
            )
            w3_weight_list = torch.stack(
                [ffn.experts[expert_idx].gate_proj.weight * frequency_tensor[expert_idx] for expert_idx in expert_indices], dim=0
            )

            # Normalize the weights by their sum
            total_weight = torch.sum(frequency_tensor[expert_indices])
            w1_weight = torch.sum(w1_weight_list, dim=0) / (total_weight )
            w2_weight = torch.sum(w2_weight_list, dim=0) / (total_weight )
            w3_weight = torch.sum(w3_weight_list, dim=0) / (total_weight )

            # Set the merged weight to the first expert in the group
            ffn.experts[expert_indices[0]].up_proj.weight.copy_(w1_weight)
            ffn.experts[expert_indices[0]].down_proj.weight.copy_(w2_weight)
            ffn.experts[expert_indices[0]].gate_proj.weight.copy_(w3_weight)

            # Bind all experts in the group to the first expert (sharing parameters)
            for expert_idx in expert_indices[1:]:
                ffn.experts[expert_idx] = ffn.experts[expert_indices[0]]
            print(expert_indices[0])
    
    return ffn
def generate_limit_array(num_layers, num_experts, percent):
    """
    生成一个满足要求的 limit 数组。
    
    Args:
        num_layers (int): 总层数。
        num_experts (int): 专家的数量。
        percent (float): 百分比，用于计算目标平均值。
        
    Returns:
        list: 满足条件的 limit 数组。
    """
    # 计算目标平均值
    target_avg = num_experts * percent
    total_sum = target_avg * num_layers  # limit 数组的总和

    # 随机生成满足条件的数组
    limit = np.random.randint(1, target_avg*2, num_layers)
    
    # 调整数组的总和为 total_sum
    current_sum = sum(limit)
    adjustment = total_sum - current_sum
    limit = np.array(limit, dtype=float)  # 转为浮点型以进行调整

    # 按比例调整以接近 total_sum
    limit += adjustment / num_layers

    # 确保每个值是整数且 > 0
    limit = np.round(limit).astype(int)
    limit[limit < 1] = 1  # 确保所有值都至少为 1

    # 再次调整以确保总和与 total_sum 匹配
    final_adjustment = total_sum - sum(limit)
    for i in range(abs(int(final_adjustment))):
        index = i % num_layers
        limit[index] += 1 if final_adjustment > 0 else -1

    return sorted(limit, reverse=True)
def assign_experts_to_groups_by_similarity(
        model,
        frequency_list,
        expert_output, 
) :
    """
    根据专家的原始频率信息，选择每层中最大频率的专家作为定点，
    并将其他专家与定点专家进行相似度比较，按照相似度超过0.7来分组。

    Args:
        model (PhiMoEForCausalLM): The model instance.
        frequency_list (list of list of float): 32x16 的列表，表示32层中每层16个专家的频率。

    Returns:
        group (list of list of int): 32x16 的列表，表示每层16个专家所属的组索引。
    """
    num_layers = len(frequency_list)
    num_experts = len(frequency_list[0])  # 每层有多少个专家
    group = [[-1 for _ in range(num_experts)] for _ in range(num_layers)]  # 初始化组标签
    limits = generate_limit_array(num_layers, num_experts, args.percent)

    # 遍历每一层
    for layer_idx in range(num_layers):
        limit = limits[layer_idx]
        layer_frequencies = frequency_list[layer_idx]
        layer_expert_output = expert_output[layer_idx * num_experts:(layer_idx + 1) * num_experts]

        # 初始化未分组专家列表
        ungrouped_experts = list(range(num_experts))
        current_group_idx = 0

        # 分组逻辑
        while ungrouped_experts and current_group_idx < limit:
            # 找出当前组的锚点专家
            anchor_expert_idx = max(ungrouped_experts, key=lambda idx: layer_frequencies[idx])
            group[layer_idx][anchor_expert_idx] = current_group_idx
            ungrouped_experts.remove(anchor_expert_idx)

            # 计算相似度并分配到当前组
            anchor_expert_output = layer_expert_output[anchor_expert_idx]
            for expert_idx in list(ungrouped_experts):  # 使用副本，避免修改列表时出错
                current_expert_output = layer_expert_output[expert_idx]
                cos_sim = cosine_similarity(
                    np.array(anchor_expert_output).reshape(1, -1),
                    np.array(current_expert_output).reshape(1, -1)
                )[0][0]

                # 如果相似度满足条件，分配到当前组
                if cos_sim >= 0.7:
                    group[layer_idx][expert_idx] = current_group_idx
                    ungrouped_experts.remove(expert_idx)

            # 更新到下一个组
            current_group_idx += 1

        # 如果还有未分组的专家，将它们分配到最后一组
        for expert_idx in ungrouped_experts:
            group[layer_idx][expert_idx] = limit - 1

    return group

    
def merge_by_groups_with_usage_frequency_weighting(
        model,
        frequency_list,
    expert_output, 
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
    group = assign_experts_to_groups_by_similarity(model, frequency_list, expert_output)
    print(group)
    print("1")
    # 遍历每一层进行专家合并
    for layer_idx in range(1, 24):
        # 获取该层的组索引和频率信息
        group_layer = group[layer_idx]  # 获取当前层的组信息
        frequency_layer = frequency_list[layer_idx]  # 获取该层专家的频率信息
        print(layer_idx)
        # 合并当前层的专家
        model.model.layers[layer_idx].mlp = _merge_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].mlp,
            group=group_layer,
            frequency_list=frequency_layer,
        )
        print("done!")
    print("all done!")
    return model
    
def map_to_range(data, new_min=0.1, new_max=0.9):
    # 定义函数，将每个小列表映射到[0.1, 0.9]区间
    result = []
    
    for gate, values in data.items():
        old_min = min(values)
        old_max = max(values)
        # 遍历每个值进行归一化
        normalized_values = [
            0 if math.isnan(value) else value
            for value in values
        ]
        result.append(normalized_values)
    
    return result

def get_model_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    # Convert bytes to megabytes (MB)
    total_size_mb = total_size / (1024 ** 3)
    return total_size_mb
    
frequence_dict = {}



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", trust_remote_code=True)
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

model = HFLM(pretrained="/root/autodl-tmp/lm-evaluation-harness/Qwen/Qwen1.5-MoE-A2.7B", trust_remote_code=True,device = 'cpu')
numm = 0
for name, layer in model._model.named_modules():
    if 'mlp.gate' in name:
        layer.register_forward_hook(hook_fn)
        numm += 1
        # print(name,layer)
        # break
    if 'down_proj' in name and 'experts' in name:
        handle = layer.register_forward_hook(hook_fn_expert)
        hook_handles.append(handle)
        
print(f'all_gate number:{numm}')
results = lm_eval.simple_evaluate( # call simple_evaluate
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
frequency_list = map_to_range(captured_outputs)
expert_output = map_to_range(captured_expert_output)
modell = merge_by_groups_with_usage_frequency_weighting(model._model, frequency_list, expert_output)
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
results = lm_eval.simple_evaluate( # call simple_evaluate
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