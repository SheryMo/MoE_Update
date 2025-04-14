#!/bin/bash

# # 定义 dataset 列表 "sciq" "tinyTruthfulQA"
# datasets=("winogrande" )

# # 遍历每个 dataset 和 percent
# for dataset in "${datasets[@]}"; do
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_full.py --dataset "$dataset" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Full_"$dataset".log 2>&1
#   for percent in $(seq 0.6 0.1 0.9); do
#     echo "Running: python script.py --dataset $dataset --percent $percent"
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_OurLayerCross_"$dataset"_"$percent".log 2>&1
#     # python Qwen_expert_merge_outSim.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Sim_"$dataset"_"$percent".log 2>&1
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Our_"$dataset"_"$percent".log 2>&1
#     # python Qwen_expert_merge_random.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_"$percent".log 2>&1
#   done
#   for percent in $(seq 0.1 0.1 0.9); do
#     # python Qwen_expert_merge_our.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Our_"$dataset"_"$percent".log 2>&1
#     python Qwen_expert_merge_random.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_"$percent".log 2>&1
#   done
# done

# datasetss=( "lambada")
# # 遍历每个 dataset 和 percent
# for dataset in "${datasetss[@]}"; do
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_full.py --dataset "$dataset" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Full_"$dataset".log 2>&1
#   for percent in $(seq 0.1 0.1 0.1); do
#     echo "Running: python script.py --dataset $dataset --percent $percent 1"
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_OurLayerCross_"$dataset"_"$percent".log 2>&1
#   done
#   for percent in $(seq 0.1 0.1 0.9); do
#     echo "Running: python script.py --dataset $dataset --percent $percent"
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_outSim.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Sim_"$dataset"_"$percent".log 2>&1
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Our_"$dataset"_"$percent".log 2>&1
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_random.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_"$percent".log 2>&1
#   done
# done "winogrande"

# datasetss=("winogrande" "sciq" "lambada")
# # 遍历每个 dataset 和 percent
# for dataset in "${datasetss[@]}"; do
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_full.py --dataset "$dataset" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Full_"$dataset".log 2>&1
#   for percent in $(seq 0.1 0.1 0.6); do
#     echo "Running: python script.py --dataset $dataset --percent $percent"
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_outSim.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Sim_"$dataset"_"$percent".log 2>&1
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_only_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_only_layerCross_"$dataset"_"$percent".log 2>&1
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_random.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_"$percent".log 2>&1
#   done
# done

# datasetss=( "truthfulqa")
# # 遍历每个 dataset 和 percent
# for dataset in "${datasetss[@]}"; do
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_full.py --dataset "$dataset" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Full_"$dataset".log 2>&1"truthfulqa"
#   for percent in $(seq 0.1 0.1 0.1); do
#     echo "Running: python script.py --dataset $dataset --percent $percent"
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_outSim.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Sim_"$dataset"_"$percent".log 2>&1 
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_only_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_only_layerCross_"$dataset"_"$percent".log 2>&1 
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Our_"$dataset"_"$percent".log 2>&1 
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_OurLayerCross_"$dataset"_"$percent".log 2>&1 
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_random.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_"$percent".log 2>&1 
    
#   done

# done

# datasetss=( "multirc")
# # 遍历每个 dataset 和 percent
# for dataset in "${datasetss[@]}"; do
#     # HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_full.py --dataset "$dataset" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Full_"$dataset".log 2>&1
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_random.py --dataset "$dataset" --percent 0.1 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_0.1.log 2>&1 
#   for percent in $(seq 0.2 0.1 0.6); do
#     echo "Running: python script.py --dataset $dataset --percent $percent"
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_outSim.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Sim_"$dataset"_"$percent".log 2>&1 
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_only_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_only_layerCross_"$dataset"_"$percent".log 2>&1 
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Our_"$dataset"_"$percent".log 2>&1 
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_OurLayerCross_"$dataset"_"$percent".log 2>&1 
#     HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_random.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_"$percent".log 2>&1 
    
#   done

# done

datasetss=("xquad")
# 遍历每个 dataset 和 percent
for dataset in "${datasetss[@]}"; do
    HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_full.py --dataset "$dataset" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Full_"$dataset".log 2>&1
  for percent in $(seq 0.1 0.1 0.6); do
    echo "Running: python script.py --dataset $dataset --percent $percent"
    HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_outSim.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Sim_"$dataset"_"$percent".log 2>&1 
    HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_only_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_only_layerCross_"$dataset"_"$percent".log 2>&1 
    HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Our_"$dataset"_"$percent".log 2>&1 
    HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_our_layerCross.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_OurLayerCross_"$dataset"_"$percent".log 2>&1 
    HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_random.py --dataset "$dataset" --percent "$percent" > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Random_"$dataset"_"$percent".log 2>&1 
    
  done

done
# HF_ENDPOINT=https://hf-mirror.com python A_expert_merge_our_layerCross.py --dataset "copa" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_Google_0413/outputQwen_layerCross_"copa".log 2>&1 
# HF_ENDPOINT=https://hf-mirror.com python Qwen_expert_merge_full.py --dataset "logiqa" --percent 0.5 > /root/autodl-tmp/lm-evaluation-harness/logs_qwen_0129/outputQwen_Full_"logiqa".log 2>&1 

# HF_ENDPOINT=https://hf-mirror.com python A_expert_merge_our_layerCross.py --dataset "winogrande" --percent 0.6 > /root/autodl-tmp/lm-evaluation-harness/logs_Google_0413/outputswitch_layerCross_wino_0.5.log 2>&1