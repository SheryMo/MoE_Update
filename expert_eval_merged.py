import lm_eval
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
import torch

# 初始化一个全局字典来保存捕获的router_logits
captured_outputs = {}
captured_one = {}
layer_count = 0

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
    group =  [[1, 0, 1, 2, 0, 1, 0, 0, 1, 2, 2, 1, 1, 0, 2, 0], [0, 2, 1, 1, 1, 0, 2, 2, 1, 2, 0, 0, 0, 1, 0, 1], [1, 1, 1, 2, 0, 0, 1, 2, 0, 1, 0, 2, 1, 0, 2, 0], [2, 0, 1, 1, 3, 2, 1, 1, 2, 0, 0, 0, 0, 1, 1, 0], [2, 0, 1, 1, 1, 1, 2, 2, 1, 0, 1, 0, 0, 2, 0, 0], [1, 0, 1, 2, 1, 1, 2, 0, 2, 0, 1, 2, 0, 0, 1, 0], [3, 1, 0, 1, 0, 1, 1, 3, 0, 1, 0, 2, 0, 1, 0, 2], [1, 1, 2, 1, 0, 1, 0, 0, 0, 2, 1, 2, 1, 2, 0, 0], [0, 2, 1, 2, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 2], [1, 0, 3, 0, 0, 3, 1, 2, 1, 0, 0, 2, 0, 1, 1, 1], [0, 1, 2, 1, 1, 0, 2, 1, 0, 0, 2, 0, 2, 1, 1, 0], [0, 3, 0, 2, 1, 1, 2, 0, 1, 0, 1, 1, 1, 2, 0, 0], [3, 1, 0, 2, 0, 1, 0, 1, 1, 2, 1, 1, 2, 0, 0, 0], [1, 2, 1, 1, 2, 1, 0, 0, 0, 1, 0, 0, 2, 0, 1, 3], [0, 0, 2, 3, 0, 0, 2, 0, 1, 2, 1, 1, 1, 0, 1, 1], [0, 0, 1, 2, 1, 2, 1, 2, 0, 1, 0, 2, 1, 0, 1, 0], [2, 1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 2, 1], [1, 0, 2, 0, 2, 2, 0, 2, 1, 1, 0, 0, 1, 1, 1, 0], [2, 1, 1, 1, 0, 1, 1, 2, 0, 3, 1, 0, 0, 0, 0, 2], [2, 1, 3, 1, 2, 0, 0, 2, 0, 0, 1, 1, 0, 1, 1, 0], [3, 0, 0, 0, 0, 2, 1, 0, 2, 0, 1, 2, 1, 1, 1, 1], [0, 0, 1, 0, 2, 1, 1, 0, 2, 2, 1, 0, 0, 1, 2, 1], [2, 1, 0, 3, 1, 0, 0, 2, 0, 1, 2, 0, 1, 0, 1, 1], [0, 1, 3, 1, 0, 2, 1, 0, 0, 2, 0, 2, 1, 1, 1, 0], [1, 1, 0, 2, 1, 0, 2, 1, 1, 0, 0, 2, 0, 3, 1, 0], [1, 1, 1, 0, 0, 0, 0, 1, 1, 3, 2, 2, 0, 2, 1, 0], [1, 1, 0, 1, 0, 0, 2, 0, 1, 1, 0, 2, 2, 0, 2, 1], [0, 0, 2, 1, 1, 0, 0, 2, 1, 0, 2, 1, 1, 0, 1, 2], [1, 0, 1, 2, 0, 1, 0, 1, 2, 2, 1, 0, 0, 0, 1, 2], [1, 0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 3, 1, 0, 2, 2], [0, 1, 0, 3, 0, 2, 1, 0, 2, 1, 1, 0, 1, 2, 0, 1], [1, 1, 1, 0, 2, 1, 3, 0, 1, 0, 1, 0, 0, 3, 0, 2]]
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
    print(f"Layer: {layer_name} - Captured router_logits: {expert_values_list}")
    print(f"Layer: {layer_name} - One-hot representation: {one_hot_list}")
    layer_count += 1
    layer_count = layer_count % 32


tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)
modell = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/lm-evaluation-harness/phiMergedMoE",
    trust_remote_code=True
)
modell = merge_by_groups_with_usage_frequency_weighting(modell)
model = HFLM(pretrained=modell,tokenizer = tokenizer, trust_remote_code=True, device="cuda")
# numm = 0
# for name, layer in model._model.named_modules():
#     if 'gate' in name:
#         layer.register_forward_hook(hook_fn)
#         numm += 1
#         # print(name,layer)
#         # break
# print(f'all_gate number:{numm}')
    
#     # print(model._model)

# for name, layer in model._model.named_modules():
#     print(name)
#     print(layer)

# indexes all tasks from the `lm_eval/tasks` subdirectory.
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
task_manager = lm_eval.tasks.TaskManager()

# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=model,
    tasks=["mmlu_formal_logic"],
    num_fewshot=5,
    task_manager=task_manager,
    batch_size = 1,
    limit = 50,
)
print(captured_outputs)
print(captured_one)
print(results['results'])