import lm_eval
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, List
import random
# 初始化一个全局字典来保存捕获的router_logits
captured_outputs = {}
captured_one = {}
layer_count = 0

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
                [ffn.experts[expert_idx].w1.weight * frequency_tensor[expert_idx] for expert_idx in expert_indices], dim=0
            )
            w2_weight_list = torch.stack(
                [ffn.experts[expert_idx].w2.weight * frequency_tensor[expert_idx] for expert_idx in expert_indices], dim=0
            )
            w3_weight_list = torch.stack(
                [ffn.experts[expert_idx].w3.weight * frequency_tensor[expert_idx] for expert_idx in expert_indices], dim=0
            )

            # Normalize the weights by their sum
            total_weight = torch.sum(frequency_tensor[expert_indices])
            w1_weight = torch.sum(w1_weight_list, dim=0) / (total_weight + 1e-6)
            w2_weight = torch.sum(w2_weight_list, dim=0) / (total_weight + 1e-6)
            w3_weight = torch.sum(w3_weight_list, dim=0) / (total_weight + 1e-6)

            # Set the merged weight to the first expert in the group
            ffn.experts[expert_indices[0]].w1.weight.copy_(w1_weight)
            ffn.experts[expert_indices[0]].w2.weight.copy_(w2_weight)
            ffn.experts[expert_indices[0]].w3.weight.copy_(w3_weight)

            # Bind all experts in the group to the first expert (sharing parameters)
            for expert_idx in expert_indices[1:]:
                ffn.experts[expert_idx] = ffn.experts[expert_indices[0]]
            print(expert_indices[0])
    
    return ffn

def assign_experts_to_groups_by_similarity(
        model,
        frequency_list,
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
    num_experts = len(frequency_list[0])  # 每层有16个专家
    group = [[-1] * num_experts for _ in range(num_layers)]  # 初始化组标签
    limit = 6
    # 遍历每一层
    for layer_idx in range(num_layers):
        if layer_idx > num_layers-10:
            limit = 2
        layer_frequencies = frequency_list[layer_idx]
        
        
        # 取出该层专家的参数
        experts_params = model.model.layers[layer_idx].block_sparse_moe.experts.parameters()
        experts_params = list(experts_params)  # 转化为列表形式
        
        # 初始化组
        current_group_idx = -1
        # 记录当前层的所有分组情况，找出未分组的专家
        ungrouped_experts = list(range(num_experts))
        
        while ungrouped_experts:
            # 选择当前未分组的最大频率专家作为定点
            anchor_expert_idx = max(ungrouped_experts, key=lambda idx: layer_frequencies[idx])
            anchor_expert_param = model.model.layers[layer_idx].block_sparse_moe.experts[anchor_expert_idx].w3.weight
            
            current_group_idx +=1
            if current_group_idx == limit:
                break
            group[layer_idx][anchor_expert_idx] = current_group_idx  # 将定点专家归为第一个组
            
            # 从未分组专家中移除定点专家
            ungrouped_experts.remove(anchor_expert_idx)
            numm = 0
            cos_simi = [-1 for _ in range(num_experts)]
            # 遍历其他未分组专家，与定点专家计算相似度
            for expert_idx in ungrouped_experts[:]:
                current_expert_param = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx].w3.weight
                cos_sim = cosine_similarity(
                    current_expert_param.to(torch.float32).view(1, -1).cpu().detach().numpy(),
                    anchor_expert_param.to(torch.float32).view(1, -1).cpu().detach().numpy()
                )[0][0]
                print(cos_sim)
                cos_simi[expert_idx] = random.random()
            for expert_idx in ungrouped_experts[:]:
                # 如果相似度超过0.7，且当前组数少于4，将其归为当前组
                if current_group_idx < limit and numm <= 4:
                    expert_idx = np.argmax(cos_simi)
                    if cos_simi[expert_idx] < 0.2:
                        break
                    group[layer_idx][expert_idx] = current_group_idx
                    cos_simi[expert_idx] = -1
                    ungrouped_experts.remove(expert_idx)  # 将已分组专家从未分组列表中移除
                    numm += 1
                    
                elif numm > 5:
                    if cos_simi[np.argmax(cos_simi)] > 0.7:
                        expert_idx = np.argmax(cos_simi)
                        group[layer_idx][expert_idx] = current_group_idx
                        cos_simi[expert_idx] = -1
                        ungrouped_experts.remove(expert_idx)  # 将已分组专家从未分组列表中移除
                        numm += 1
                
        # 如果有专家未分组，则将其归为第四组
        if ungrouped_experts:
            for expert_idx in ungrouped_experts:
                group[layer_idx][expert_idx] = limit-1
            ungrouped_experts.clear()  # 清空未分组专家列表
                    
    return group
    
def merge_by_groups_with_usage_frequency_weighting(
        model,
        frequency_list,
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
    group = assign_experts_to_groups_by_similarity(model, frequency_list)
    print(group)
    print("1")
    # 遍历每一层进行专家合并
    for layer_idx in range(32):
        # 获取该层的组索引和频率信息
        group_layer = group[layer_idx]  # 获取当前层的组信息
        frequency_layer = frequency_list[layer_idx]  # 获取该层专家的频率信息
        print(layer_idx)
        # 合并当前层的专家
        model.model.layers[layer_idx].block_sparse_moe = _merge_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].block_sparse_moe,
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
            new_min + ((value - old_min) / (old_max - old_min)) * (new_max - new_min)
            for value in values
        ]
        result.append(normalized_values)
    
    return result

frequence_dict = {'gate_0': [10.43655577301979, 12.285117015242577, 12.643037177622318, -30.57570916414261, -26.939604230225086, -13.9039960000664, 14.65974435210228, -17.031257734633982, 9.50034423917532, 10.471574991941452, 11.43256390094757, 9.191953528672457, 10.98190937191248, 12.938804179430008, -129.7072583436966, 14.323790952563286], 'gate_1': [10.893136478960514, 6.678319530561566, 4.510218909010291, 5.906390832737088, 9.711199190467596, 4.275940561201423, 6.381633393466473, 9.116557661443949, 2.5423965050722472, 8.78019766509533, -22.972466230392456, 6.471608951687813, 2.2201348706439603, -2.8207746051921276, 2.6125371058878954, 3.200483025982976], 'gate_2': [10.05173933878541, 7.865251369774342, 11.737056352198124, 8.605761347338557, 13.680877469480038, 12.805180668830872, 8.839563518762589, -9.960745207965374, 10.289375387132168, 10.902438906021416, 2.3798921764973784, 11.142438560724258, -21.439139492809772, 3.742857208424539, -3.1633864770410582, 13.880326971411705], 'gate_3': [16.450524240732193, 20.199453622102737, 16.59987334907055, 18.24657978117466, 16.77475231140852, 15.981928631663322, 4.477095305046532, 23.458605960011482, 22.79249081015587, -58.52380168437958, 1.8602427792502567, 12.583093672990799, 24.459719821810722, -28.64323753118515, -2.848898929893039, -5.992663963814266], 'gate_4': [10.040008094161749, 13.194371581077576, 12.086394861340523, 16.167471577151446, 7.88672100007534, -7.132705230033025, 5.5907903698971495, 6.187464457936585, -8.897794169577537, 2.782798212778289, -20.388100296258926, 18.266727209091187, -5.097018745269452, -22.48734661191702, 16.345801033079624, -5.709401144646108], 'gate_5': [8.612545735901222, 17.886975325644016, 8.795624578371644, 14.443028568755835, -10.117786338581936, -3.3952711005404126, 15.80876437301049, 2.9877364145650063, 16.52306817471981, -9.137731119059026, -9.662854617461562, 12.171279612928629, -20.935045678168535, 18.241725228726864, 17.504020757973194, 9.884813636541367], 'gate_6': [14.707765348255634, 4.719553906208603, 18.20614193379879, 42.8145694732666, 22.946811884641647, 17.931309185922146, -12.086248174222419, -5.826608587987721, 45.668201848864555, 23.15218634903431, -59.12982687354088, 40.431214064359665, 27.810424581170082, -24.354783153161407, 24.028709808830172, 23.440494067966938], 'gate_7': [46.07598511874676, 31.993840724229813, 34.72982630133629, 18.664920926094055, -98.16770207881927, 40.93697239458561, 25.787006855010986, -29.73519482697884, 53.326725997030735, 21.797449864912778, -15.358635138371028, 33.17673970758915, 33.14234238117933, 24.795929051935673, -60.69492529332638, -22.36123051540926], 'gate_8': [33.994922786951065, 40.41843627393246, -25.53022559452802, 41.725864350795746, 61.61094503104687, 45.13979463279247, 34.641305670142174, -26.837450795108452, 23.405520491302013, 83.63105589151382, 12.365640857780818, 30.051815200597048, 56.75871151685715, -49.3216841458343, -47.02785921189934, 40.37649239599705], 'gate_9': [59.73922681808472, 35.69058337435126, 59.18547251820564, 109.60132640600204, 54.507130831480026, 38.18960744142532, 82.15521749854088, 73.53096655011177, 81.91620667278767, 24.917668405221775, 43.9592601954937, 88.90635174512863, 101.26820421218872, 36.438099099788815, 94.27259331941605, 88.78682500123978], 'gate_10': [129.36747348308563, 111.35751760005951, 78.52138012647629, 100.75830227136612, 119.40734845399857, 74.89235171675682, 86.4811259508133, 91.52522224187851, 37.65089261403773, 80.4830790758133, 95.7199736237526, 114.93933749198914, 65.25584926456213, 34.803542494773865, 105.2508379817009, 62.30615943670273], 'gate_11': [127.32808446884155, 145.00946003198624, 120.24786907434464, 155.26697927713394, 173.98233783245087, 67.53318287432194, 120.76962381601334, 133.79275023937225, 129.43162834644318, 175.53035736083984, 119.36109739542007, 134.6842595934868, 71.8437709659338, 77.23222211003304, 14.187465987750329, 107.61515128612518], 'gate_12': [37.83478996963913, 118.01651531457901, 155.8667813539505, 148.7983351945877, 65.07458370178938, 91.83594918251038, 133.44337844848633, 150.64085084199905, 87.38372525572777, 59.74739155173302, 113.36565655469894, 115.34889620542526, 121.89785915613174, 113.41711378097534, 91.63387402892113, 127.94984072446823], 'gate_13': [115.82806128263474, 127.44042557477951, 144.63836652040482, 80.51140277087688, 147.90671283006668, 9.839900693390518, 149.4501270055771, 258.2918031215668, 137.2469503879547, 131.97855788469315, 167.75203520059586, 170.34881246089935, 31.082730429479852, 126.54411596059799, 152.7592608332634, 104.97116687893867], 'gate_14': [100.8526548743248, 105.96287643909454, 125.3211452960968, 114.27149724960327, 133.18333852291107, 129.86056792736053, 60.43849143292755, 157.6385633945465, 101.6737768650055, 116.34082460403442, -12.259191949851811, 104.43720635771751, 130.9246277809143, 141.64558655023575, 148.69793820381165, 81.01122218370438], 'gate_15': [65.9192861020565, 155.18381768465042, 136.05421829223633, 124.60115593671799, 105.32289999723434, 122.86085134744644, 168.0888838171959, 135.67287695407867, 124.19023019075394, 133.94590878486633, 100.40092474222183, -25.32335533760488, 136.9910204410553, 118.83086150884628, 126.73351180553436, 179.02699494361877], 'gate_16': [96.93212449550629, 33.78248275857186, 110.3131445646286, 100.76236283779144, 91.46257400512695, 68.25287318229675, 43.817765055340715, 112.85054636001587, -97.14076210930943, 292.71310901641846, -90.82782900333405, 83.63295704126358, 114.0223628282547, 116.78943109512329, 109.26954835653305, 52.45431824028492], 'gate_17': [27.84545379143674, 102.1059921681881, 186.34132194519043, 176.03286284208298, 187.586172580719, 187.83268690109253, 183.48128807544708, 186.99897134304047, 225.65341758728027, 187.55882847309113, 272.1367870569229, 189.17323851585388, 180.50547790527344, 164.23128616809845, 88.24927744269371, 178.99513018131256], 'gate_18': [192.99291670322418, 173.470099568367, 194.96224069595337, 225.57137823104858, 137.2833097577095, 168.06553786993027, 237.8670346736908, 229.96018075942993, 288.8437201976776, 191.46557581424713, 163.08054274320602, 130.82149028778076, 170.43194818496704, 151.8617160320282, 195.2157506942749, 196.43896412849426], 'gate_19': [193.0746853351593, 176.92400753498077, 185.5810351371765, 177.26472544670105, 205.6779670715332, 142.36358177661896, 169.9977849125862, 174.40388095378876, 243.5815944671631, 190.71863389015198, 207.01616883277893, 195.24583792686462, 253.04887104034424, 199.46099746227264, 192.50930452346802, 190.21448600292206], 'gate_20': [136.42129170894623, 142.8191316127777, 137.01980966329575, 123.8697424530983, 13.243654631367463, 167.298912525177, 156.9219331741333, 226.3077986240387, 110.29608601331711, 162.41977524757385, 164.55216217041016, 144.9320297241211, 146.03904736042023, 175.79514634609222, 133.76057213544846, 70.4017224162817], 'gate_21': [270.51507461071014, 289.3117940425873, 274.31037187576294, 259.13108944892883, 257.1290498971939, 273.0370478630066, 256.07913959026337, 282.9588311910629, 257.78192377090454, 277.5027128458023, 302.661563873291, 336.7664623260498, 234.18157041072845, 310.53259897232056, 286.53676414489746, 276.76399993896484], 'gate_22': [191.2687268257141, 180.3163390159607, 242.80707573890686, 147.43834257125854, 196.84582448005676, 206.70305740833282, 179.40140235424042, 185.55196833610535, 199.11408245563507, 213.52936100959778, 201.05712020397186, 266.58742678165436, 201.10766458511353, 205.74341690540314, 171.14352756738663, 218.94103813171387], 'gate_23': [301.57435297966003, 229.8403217792511, 159.00534456968307, 188.79722845554352, 204.074502825737, 193.47986161708832, 186.65158486366272, 189.0693517923355, 195.43895554542542, 186.4992152452469, 199.29718494415283, 192.95154345035553, 173.67957735061646, 109.09274247288704, 182.12182676792145, 198.44780158996582], 'gate_24': [300.8309180736542, 306.9479932785034, 291.50154423713684, 279.6400992870331, 339.8325786590576, 296.12316036224365, 323.5502734184265, 283.6120744943619, 282.0226607322693, 290.8795071840286, 340.6225402355194, 284.3483191728592, 294.1676650047302, 312.12233328819275, 301.2096059322357, 302.7523822784424], 'gate_25': [213.4820773601532, 218.20799720287323, 197.02273881435394, 223.28713834285736, 177.2047120332718, 202.3722882270813, 209.5419499874115, 244.57658755779266, 210.21045994758606, 201.982235789299, 207.10834109783173, 194.17071986198425, 206.31709706783295, 191.6612708568573, 226.44749748706818, 250.44800889492035], 'gate_26': [232.51770496368408, 225.31877064704895, 220.0389734506607, 224.75741243362427, 218.72060179710388, 228.48564839363098, 219.11324560642242, 221.8437852859497, 227.1184619665146, 239.10245776176453, 281.64393866062164, 211.72109735012054, 218.65574717521667, 234.14996123313904, 229.98712372779846, 225.31604838371277], 'gate_27': [201.7092660665512, 208.6942048072815, 204.6682459115982, 215.63237726688385, 201.20955514907837, 200.92854762077332, 220.73446118831635, 213.92966413497925, 202.76720690727234, 216.13627970218658, 177.4655681848526, 208.5226525068283, 207.39869701862335, 232.24456644058228, 203.0385843515396, 204.44910776615143], 'gate_28': [262.6080673933029, 271.18580543994904, 310.35455083847046, 260.5741186141968, 271.1687433719635, 264.7134962081909, 316.3416357040405, 274.75066471099854, 261.43794083595276, 261.67192482948303, 256.4815766811371, 241.33479022979736, 255.89150857925415, 242.95710492134094, 294.60696387290955, 265.0441623926163], 'gate_29': [659.3155002593994, 701.2020802497864, 654.6063270568848, 647.6133470535278, 647.2977514266968, 640.6768069267273, 659.4208550453186, 687.916356086731, 682.1207489967346, 657.3825387954712, 653.9346632957458, 642.3912782669067, 661.1935629844666, 662.840922832489, 676.9284625053406, 653.0358362197876], 'gate_30': [487.69743251800537, 470.8118906021118, 466.09358978271484, 449.8996171951294, 470.77901697158813, 472.89546823501587, 446.68706345558167, 475.0325779914856, 481.188597202301, 476.0990288257599, 460.0677812099457, 423.85263657569885, 462.8587098121643, 470.07510900497437, 478.4903974533081, 483.47697353363037], 'gate_31': [357.7453718185425, 369.49005699157715, 358.0618453025818, 338.2703547477722, 349.20234537124634, 364.40229749679565, 351.8476140499115, 334.088219165802, 353.16279888153076, 345.10497307777405, 390.92642402648926, 283.67029535770416, 358.9190921783447, 349.580913066864, 393.49659848213196, 358.88091135025024]}

frequency_list = map_to_range(frequence_dict)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)

model = HFLM(pretrained="/root/autodl-tmp/lm-evaluation-harness/microPhi35MoE", trust_remote_code=True, device='cpu')

modell = merge_by_groups_with_usage_frequency_weighting(model._model, frequency_list)

print("begin to save!")
modell.save_pretrained('/root/autodl-tmp/lm-evaluation-harness/phiMergedMoESS')
# # numm = 0
# # for name, layer in model._model.named_modules():
# #     if 'gate' in name:
# #         layer.register_forward_hook(hook_fn)
# #         numm += 1
# #         # print(name,layer)
# #         # break
# # print(f'all_gate number:{numm}')
    
#     # print(model._model)

# # for name, layer in model._model.named_modules():
# #     if 'gate' in name:
# #         print(name,layer)
# #         break

# # indexes all tasks from the `lm_eval/tasks` subdirectory.
# # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# # to include a set of tasks in a separate directory.
# task_manager = lm_eval.tasks.TaskManager()

# # Setting `task_manager` to the one above is optional and should generally be done
# # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
# results = lm_eval.simple_evaluate( # call simple_evaluate
#     model=model,
#     tasks=["mmlu_formal_logic"],
#     num_fewshot=0,
#     task_manager=task_manager,
# )
# print(captured_outputs)
# print(captured_one)
# print(results['results'])