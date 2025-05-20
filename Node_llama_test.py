import lm_eval
import numpy as np
import math
import argparse
from sklearn.metrics.pairwise import cosine_similarity
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, SwitchTransformersForConditionalGeneration
from lm_eval.models.huggingface import HFLM
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, List
import random
import torch.nn.functional as F  # 引入softmax函数
from task_util import *
import os
import time
import json
import hashlib
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import copy
import gc
import logging
import queue
from multiprocessing import Semaphore, Process
import multiprocessing
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
global_node_infor = {}

def run_node_inference_with_limit(node, semaphore):
    with semaphore:
        try:
            node.model_inference_and_update_loop()
        except Exception as e:
            print(f"[Process {node.ip}] Error during inference: {e}")

def start_nodes_with_limited_concurrency(nodes, max_concurrent=2):
    semaphore = Semaphore(max_concurrent)
    processes = []

    for node in nodes:
        p = Process(target=run_node_inference_with_limit, args=(node, semaphore))
        p.start()
        processes.append(p)

    return processes
    
logger = logging.getLogger("save_model")
logger.setLevel(logging.DEBUG)
########################################################################################
# 初始化参数解析器
parser = argparse.ArgumentParser(description="Dataset selection for processing")
# 添加参数
parser.add_argument('--dataset', type=str, default="winogrande", help="Specify the dataset to use")
parser.add_argument('--percent', type=float, default="0.5", help="merge parts is x of the whole one")

# 解析命令行参数
args = parser.parse_args()
args.percent = 0.3  # Set the 'percent' argument manually
############################################################################
# Node类：定义节点行为
class Node:
    def __init__(self, ip, args, task, neighbors=None, upload_folder='uploads', info_table_path='node_info.json', gpu_id=None):
        self.ip = ip  # 本地节点的IP地址
        self.args = args
        self.diff = 0.6  # 差异阈值
        self.neighbors = neighbors if neighbors else []  # 邻居节点列表
        self.local_model = None  # 模型文件的默认路径
        self.X = None  # 本地X矩阵
        self.Y = None  # 本地Y矩阵
        self.node_info = {}  # 存储收到的邻居节点信息
        self.upload_folder = "saved_models/"  # 文件存储目录
        os.makedirs(self.upload_folder, exist_ok=True)  # 创建文件存储目录
        self.node_info_table = {}
        # 消息队列
        # self.message_queue = queue.Queue()
        # threading.Thread(target=self._process_message_queue, daemon=True).start()
        
        self.gpu_id = gpu_id  # 分配的 GPU ID
        self.device = f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cpu"

        self.init_task = task
        self.task_group = None
        self.task_dict = None
        self.tokenizer = None
        self.task_manager = None
        self.model = None
        self.num_expert = 8
        self.num_layer = 32
        # 初始化捕获的输出
        self.captured_outputs = {}
        self.captured_one = {}
        self.layer_count = 0
        self.being_Information = True
        self.start_time = None
        self.update_ava = False
        self.update_solution = None
        self.model_fre = None
        self.expanded_task_list = [
            # 'AraDiCE_ArabicMMLU_high_humanities_history_egy',
            # 'AraDiCE_ArabicMMLU_high_humanities_islamic-studies_lev',
            # 'AraDiCE_piqa_egy',
            # 'AraDiCE_ArabicMMLU_high_stem_biology_egy',
            'arc_easy',
            # 'arc_challenge',
            # 'anagrams1',
            'anli_r2',
            'anli_r1',
            # 'arabic_leaderboard_arabic_mmlu_high_school_statistics_light',
            'coqa',
            'eq_bench',
            'fda',
            'cola',
            'mnli',
            'mrpc',
            'qnli',
            # 'qqp',
            'rte',
            # 'sst',
            'wnli',
            # 'gpqa_main_zeroshot',
            # 'gpqa_diamond_zeroshot',
            # 'gpqa_extended_zeroshot',
            # 'gpqa_main_n_shot',
            # 'gpqa_diamond_n_shot',
            # 'gpqa_extended_n_shot',
            # 'gpqa_main_generative_n_shot',
            # 'gpqa_diamond_generative_n_shot',
            # 'gpqa_extended_generative_n_shot',
            # 'gpqa_main_cot_zeroshot',
            # 'gpqa_diamond_cot_zeroshot',
            # 'gpqa_extended_cot_zeroshot',
            # 'gpqa_main_cot_n_shot',
            # 'gpqa_diamond_cot_n_shot',
            # 'gpqa_extended_cot_n_shot',
            # 'lambada_openai',
            # 'lambada_standard',
            # 'leaderboard_bbh_causal_judgement',
            # 'leaderboard_bbh_disambiguation_qa',
            # 'leaderboard_bbh_hyperbaton',
            # # 'leaderboard_bbh_logical_deduction_five_objects',
            # 'leaderboard_bbh_navigate',
            # # 'leaderboard_bbh_object_counting',
            # 'leaderboard_bbh_reasoning_about_colored_objects',
            # 'leaderboard_bbh_ruin_names',
            # # 'leaderboard_bbh_salient_translation_error_detection',
            # # 'leaderboard_bbh_sports_understanding',
            # 'leaderboard_bbh_temporal_sequences',
            # # 'leaderboard_bbh_tracking_shuffled_objects_seven_objects',
            # # 'leaderboard_bbh_tracking_shuffled_objects_three_objects',
            # 'leaderboard_bbh_web_of_lies', 
            'mastermind_24_easy',
            # 'mastermind_24_hard',
            'mastermind_35_easy',
            # 'mastermind_35_hard',
            'mastermind_46_easy',
            # 'mastermind_46_hard',
            'logiqa',
            # # 'mmlu',
            # 'mmlu_stem',
            # 'mmlu_humanities',
            # 'mmlu_other',
            # # 'mmlu_pro_psychology',
            # # 'mmlu_pro_physics',
            # # 'mmlu_pro_philosophy',
            # # 'mmlu_pro_other',
            # # 'mmlu_pro_math',
            # # 'mmlu_pro_law',
            # # 'mmlu_pro_history',
            # # 'mmlu_pro_health',
            # # 'mmlu_pro_engineering',
            # # 'mmlu_pro_economics',
            # # 'mmlu_pro_computer_science',
            # # 'mmlu_pro_chemistry',
            # 'mmlu_social_sciences',
            # 'openbookqa',
            # 'piqa',
            'sciq',
            'boolq',
            'cb',
            'copa',
            'multirc',
            # 'record',
            'rte',
            'wic',
            'wsc',
            # 'super_glue-boolq-t5-prompt',
            # 'super_glue-cb-t5-prompt',
            # 'super_glue-copa-t5-prompt',
            # 'super_glue-multirc-t5-prompt',
            # 'super_glue-record-t5-prompt',
            # 'super_glue-rte-t5-prompt',
            # 'super_glue-wic-t5-prompt',
            # 'super_glue-wsc-t5-prompt',
            # 'truthfulqa_mc1',
            # 'truthfulqa_mc2',
            # 'truthfulqa_gen',
            'winogrande',
            # 'wikitext'
            ]


        # self.upload_folder = upload_folder  # 文件上传目录
        self.info_table_path = info_table_path  # 节点信息存储路径
        self.zero_param = nn.Parameter(torch.zeros(1, 1).to(self.device))

        # 创建文件夹并初始化信息表
        os.makedirs(self.upload_folder, exist_ok=True)
        if not os.path.exists(self.info_table_path):
            with open(self.info_table_path, 'w') as f:
                json.dump({}, f)
                
        # # 初始化Flask应用
        # self.app = Flask(__name__)
        # self.app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 单块最大200MB
        # self._initialize_routes()
        
        self.port = 5000 + int(self.ip)
        
        # 初始化捕获的专家输出
        self.captured_expert_output = {}
        self.layer_expert_count = 0

    def merge_chunks(self, temp_dir, filename, expected_hash):
        """将分块合并并进行SHA256校验"""
        merged_path = os.path.join(self.upload_folder, filename)
        with open(merged_path, 'wb') as f_out:
            parts = sorted(os.listdir(temp_dir), key=lambda x: int(x.split('.part')[-1]))
            for part_file in parts:
                part_path = os.path.join(temp_dir, part_file)
                with open(part_path, 'rb') as f_in:
                    f_out.write(f_in.read())
    
        calculated_hash = self.calculate_sha256(merged_path)
        if calculated_hash == expected_hash:
            print(f"[SUCCESS] SHA256 verified for {filename}")
        else:
            print(f"[ERROR] SHA256 mismatch! Expected {expected_hash}, got {calculated_hash}")
    
        # 清理临时块
        for part_file in parts:
            os.remove(os.path.join(temp_dir, part_file))
        os.rmdir(temp_dir)
    
    def calculate_sha256(self, file_path, chunk_size=1024*1024):
        """计算文件的SHA256哈希值"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    
    def _initialize_routes(self):
        """初始化Flask的路由"""
        self.app.add_url_rule('/upload', 'upload_file', self.upload_file, methods=['POST'])
        self.app.add_url_rule('/download/<filename>', 'download_file', self.download_file, methods=['GET'])
        self.app.add_url_rule('/receive_info', 'receive_info', self.receive_info, methods=['POST'])

    def upload_file(self):
        """接收分块上传的文件，并在最后合并校验"""
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
    
        file = request.files['file']
        filename = request.form['filename']
        chunk_index = int(request.form['chunk_index'])
        total_chunks = int(request.form['total_chunks'])
        file_hash = request.form['file_hash']
    
        temp_dir = os.path.join(self.upload_folder, filename)
        os.makedirs(temp_dir, exist_ok=True)
    
        chunk_path = os.path.join(temp_dir, f"{filename}.part{chunk_index}")
        file.save(chunk_path)
    
        # 检查是否全部块收齐
        uploaded_chunks = os.listdir(temp_dir)
        if len(uploaded_chunks) == total_chunks:
            print(f"All chunks received for {filename}. Starting to merge...")
            self.merge_chunks(temp_dir, filename, file_hash)
    
        return jsonify({"status": "chunk received"}), 200


    def download_file(self, filename):
        """下载文件"""
        file_path = os.path.join(self.upload_folder, filename)
        if os.path.exists(file_path):
            return send_from_directory(self.upload_folder, filename)
        else:
            return jsonify({"error": "File not found"}), 404
    def receive_info(self):
        """接收信息并入队列，异步处理"""
        try:
            data = request.get_json(force=True)
            print(f"[RECEIVE] Queued message from {data.get('ip')}")
            
            # 简单结构校验
            if not data or 'ip' not in data:
                return jsonify({"error": "Invalid data"}), 400
            
            self.message_queue.put(data)
            return jsonify({"message": "Message accepted and queued"}), 200
    
        except Exception as e:
            print(f"[RECEIVE][ERROR] Exception: {e}")
            return jsonify({"error": "Internal server error"}), 500

    def _process_message_queue(self):
        while True:
            try:
                data = self.message_queue.get(timeout=5)  # 最多等待5秒
            except queue.Empty:
                # 队列五秒内没有消息，跳过或可执行其他操作
                continue
    
            try:
                print(f"[QUEUE] Processing info from {data['ip']}")
    
                required_fields = ['ip', 'hops', 'path', 'timestamp', 'X', 'Y', 'model_path']
                if any(field not in data or data[field] is None for field in required_fields):
                    print(f"[QUEUE][ERROR] Missing fields in data from {data.get('ip')}")
                    continue
    
                if self.start_time is None:
                    self.start_time = datetime.now()
    
                if not self.being_Information:
                    print(f"[QUEUE] Reject info from {data['ip']}, node updating.")
                    continue
    
                with threading.Lock():
                    if data['ip'] in self.node_info_table:
                        print(f"[QUEUE] Already have info from {data['ip']}")
                        continue
    
                    self.node_info_table[data['ip']] = {
                        'hops': data['hops'],
                        'path': data['path'],
                        'timestamp': data['timestamp'],
                        'X': data['X'],
                        'Y': data['Y'],
                        'model_path': data['model_path']
                    }
    
                print(f"[QUEUE] Stored info from {data['ip']}")
    
    
            except Exception as e:
                print(f"[QUEUE][ERROR] Failed to process message: {e}")

        
    # def receive_info(self):
    #     """接收信息并存储到本地表中"""
    
    #     try:
    #         data = request.get_json(force=True)
    #         print(f"\n[RECEIVE] Incoming data:\n{data}")
    
    #         if data is None:
    #             return jsonify({"error": "No JSON received"}), 400
    
    #         if self.start_time is None:
    #             self.start_time = datetime.now()
    
    #         if not self.being_Information:
    #             print("[RECEIVE] Rejecting info — node is being updated")
    #             return jsonify({"error": "The node is being updated, not accepted."}), 400
    
    #         # 字段完整性检查
    #         required_fields = ['ip', 'hops', 'path', 'timestamp', 'X', 'Y', 'model_path']
    #         for field in required_fields:
    #             if field not in data or data[field] is None:
    #                 return jsonify({"error": f"Missing or null field: {field}"}), 400
    
    #         # 线程安全写入 node_info_table
    #         with threading.Lock():
    #             if data['ip'] in self.node_info_table:
    #                 print(f"[RECEIVE] Node {data['ip']} already recorded. Skipping.")
    #                 return jsonify({"error": f"Node with IP {data['ip']} already exists."}), 400
    
    #             self.node_info_table[data['ip']] = {
    #                 'hops': data['hops'],
    #                 'path': data['path'],
    #                 'timestamp': data['timestamp'],
    #                 'X': data['X'],
    #                 'Y': data['Y'],
    #                 'model_path': data['model_path']
    #             }
    
    #         print(f"[RECEIVE] Stored node info from {data['ip']}")
    
    #         # 发起异步转发（安全封装）
    #         self.forward_info_to_neighbors(data)
    
    #         # 检查是否可线性组合更新
    #         self.update_ava, self.update_solution = self.check_linear_valuable()
    #         if self.update_ava:
    #             self.being_Information = False
    #             print(f"[RECEIVE] Node update is now scheduled (solution found).")
    
    #         return jsonify({"message": f"Information from {data['ip']} stored successfully!"}), 200
    
    #     except Exception as e:
    #         print(f"[RECEIVE][ERROR] Exception occurred: {e}")
    #         return jsonify({"error": "Internal server error"}), 500
    
    def extract_all_X_from_table(self,info_table_path):
        # 读取当前的节点信息表
        # with open(info_table_path, 'r') as f:
        #     node_info_table = json.load(f)

        # 创建一个列表，用于存储所有节点的 X 数据
        X_list = []
        ip_node = []
        global global_node_infor
        # 遍历所有节点的记录并提取 X
        for node_ip, node_data in global_node_infor.items():
            if 'X' in node_data:
                X_list.append(node_data['X'])  # 将 X 添加到列表中
                ip_node.append(node_ip)  # 将 IP 添加到列表中
        
        return X_list,ip_node
    
    def calculate_X(self,X):
        """处理self.X中的weights，并将其填充至一个固定长度的list"""
        result = []  # 用于存储填充后的结果
        for group_idx, group_data in X.items():
            layer = group_data['layer']
            weights = group_data['weights']

            # 如果layer的长度为1，进行填充
            if len(layer) == 1:
                a = layer[0]
                # 在最前面加上 a*self.num_expert 个 0
                padded_weights = [0] * (a * self.num_expert) + weights
                # 在后面加上 (self.num_layer - a - 1) * self.num_expert 个 0
                padded_weights += [0] * ((self.num_layer - a - 1) * self.num_expert)
            else:
                # 如果layer的长度大于1，取最小值和最大值
                a = min(layer)
                b = max(layer)

                # 在最前面加上 a*self.num_expert 个 0
                padded_weights = [0] * (a * self.num_expert) + weights
                # 在后面加上 (self.num_layer - b - 1) * self.num_expert 个 0
                padded_weights += [0] * ((self.num_layer - b - 1) * self.num_expert)
            result.append(padded_weights)  # 将填充后的结果添加到列表中

        return result

    def process_X(self, X_list):
        # 创建空的二维列表 compare_X
        compare_X = []
        ip_length = []
        # 处理 X_list 中每个元素
        for X in X_list:
            cal_X = self.calculate_X(X)
            compare_X.extend(cal_X)
            ip_length.append(len(cal_X))
        # sel_XX = self.calculate_X(self.X)
        # compare_X.extend(sel_XX)
        # ip_length.append(len(sel_XX))
        return compare_X,ip_length

    def check_linear_valuable(self):
        X_list,ip_node = self.extract_all_X_from_table(self.info_table_path)
        compare_X,ip_length = self.process_X(X_list)
        compare_Y = self.calculate_X(self.Y)
        solu = {}
        # 这里只是其他节点，X_list最后一部分还有自己本地已有的expert
        solu['ip_node'] = ip_node
        solu['ip_length'] = ip_length
        solution = []
        # print("start to check the linear valuable:")
        # print(f"compare X is : {compare_X}, and compare Y is {compare_Y}")
        # 使用NumPy进行线性组合的求解
        for y in compare_Y:
            # 将compare_X转化为NumPy矩阵
            A = np.array(compare_X)
            b = np.array(y)
            w = np.random.randn(A.shape[0])
            solution.append(w)  # 将解添加到solution列表中
        solu['solution'] = solution
        return True, solu  # 如果所有行都可以线性组合得到，返回True并且返回权重向量w
        
    def initialize_model(self):
        """初始化模型，任务组，任务字典，tokenizer"""
        # # 提取任务组和任务字典
        # self.task_group = extract_task_groups('/root/mo-e_-merge_and_-update_-mec/logs11.log')
        # self.task_dict = extract_task_dict('/root/mo-e_-merge_and_-update_-mec/logs11.log', self.task_group)

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("llama-moe/LLaMA-MoE-v1-3_5B-2_8", trust_remote_code=True)
        self.task_manager = lm_eval.tasks.TaskManager()
        print(self.ip)
        print(self.device)
        print(self.init_task)

        
        # 加载模型
        self.model = HFLM(pretrained="llama-moe/LLaMA-MoE-v1-3_5B-2_8", 
                          trust_remote_code=True ,device=self.device)
        # self.model._model = self.model._model.to_empty(self.device)

        # 注册forward hook
        self.hook_handles = []
        self.register_hooks(self.model)
        results = lm_eval.simple_evaluate( # call simple_evaluate squad_completion
            model=self.model,
            tasks=[self.init_task],
            num_fewshot=0,
            task_manager=self.task_manager,
            batch_size = 5,
            limit = 100,
            device = self.device,
        )
        print('full model:')
        print(results['results'])
        # print(self.captured_outputs)
        frequency_list,names_fre = map_to_range(self.captured_outputs,[])
        self.model_fre = frequency_list
        # self.model_fre.to('cpu')
        expert_output,names_out = map_to_range(self.captured_expert_output,[])
        # expert_output = expert_output.to('cpu')

        # 清除 forward hooks，释放模型内部引用
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        # # 明确断开引用
        # del self.model._model
        # self.model = AutoModelForCausalLM.from_pretrained('llama-moe/LLaMA-MoE-v1-3_5B-2_8', torch_dtype=torch.bfloat16, trust_remote_code=True)
        # self.model._model.to('cpu')
        modell = self.merge_by_groups_with_usage_frequency_weighting(self.model._model, self.model_fre, expert_output,names_fre)

        del results  # 评估结果也可能含有张量引用
        del self.model
        # 清除 Hook 输出
        self.captured_outputs.clear()
        self.captured_expert_output.clear()
        # 清理缓存
        
        torch.cuda.set_device(self.gpu_id)
        torch.cuda.empty_cache()
        gc.collect()
        # clean_and_report_cuda_tensors("after model release")

        # modell = modell.to('cpu')
        self.local_model = HFLM(pretrained="llama-moe/LLaMA-MoE-v1-3_5B-2_8", 
                          trust_remote_code=True ,device=self.device)
        modell.to(self.device)
        self.local_model._model = modell
        model_size = get_model_size(modell)
        print(f"Model size: {model_size:.4f} GB")
        # clean_and_report_cuda_tensors("after create model")
        del modell

        # self.start_flask_server()
        # self.start_inference_thread()

    def start_all_services(self):
        """统一启动 Flask 服务器和推理线程"""
        self.start_flask_server()
        self.start_inference_thread()
        
    def register_hooks(self,model):
        self.hook_handles = []  # 清理旧的

        for name, layer in model._model.named_modules():
            # 正确 Hook 到 TopKBalancedNoisyGate 层
            if name.endswith("mlp.gate"):
                # print(f"[HOOK] Registering gate hook on: {name}")
                handle = layer.register_forward_hook(self.hook_fn)
                self.hook_handles.append(handle)

            if 'LinearGLUExperts' in layer.__class__.__name__:
                # print(f"[HOOK] Registering expert hook on: {name}")
                handle = layer.register_forward_hook(self.hook_fn_expert)
                self.hook_handles.append(handle)
    

    def hook_fn(self, module, input, output):
        layer_name = f"gate_layers_{self.layer_count}"
        self.layer_count = (self.layer_count + 1) % 32

        # 解析 top-2 routing 结果
        topk_indices = output['topK_indices']     # shape: [tokens, 2]
        topk_scores = output['topK_scores']       # shape: [tokens, 2]
        num_experts = output['load'].shape[0]     # typically 8

        expert_score_sum = torch.zeros(num_experts)

        # 累加所有 token 的 expert 分配分数
        for i in range(topk_indices.shape[0]):
            for k in range(2):  # top-2
                expert_idx = topk_indices[i, k].item()
                expert_score = topk_scores[i, k].item()
                expert_score_sum[expert_idx] += expert_score

        # 归一化（可选）：对 token 数量求平均
        expert_score_sum /= topk_indices.shape[0]

        expert_values_list = expert_score_sum.tolist()

        # one-hot version: 只统计最常被选中的 expert
        max_value = max(expert_values_list)
        one_hot_list = [1 if v == max_value else 0 for v in expert_values_list]

        # 累加进记录中
        self.captured_outputs[layer_name] = [
            x + y for x, y in zip(self.captured_outputs.get(layer_name, [0]*num_experts), expert_values_list)
        ]
        self.captured_one[layer_name] = [
            x + y for x, y in zip(self.captured_one.get(layer_name, [0]*num_experts), one_hot_list)
        ]

    def hook_fn_expert(self, module, input, output):
        """
        Hook 每层 LinearGLUExperts，获取每个 expert 的单独输出（L2 norm），存入 self.captured_expert_output。
        跳过已经存在非 NaN 数据的层。
        """
        try:
            input_tensor = input[0]  # 从输入 tuple 拿 tensor
            layer_idx = self.layer_count
            self.layer_count = (self.layer_count + 1) % self.num_layer

            for i in range(self.num_expert):
                layer_name = f"layers_{layer_idx}_expert_{i}"

                # ✅ 如果已存在非 NaN，有效值则跳过
                if layer_name in self.captured_expert_output:
                    cached = self.captured_expert_output[layer_name]
                    if isinstance(cached, list) and len(cached) > 0 and not math.isnan(cached[0]):
                        continue

            # 获取所有 expert 的输出
            expert_outputs = module.forward_all_experts(input_tensor)

            for i, expert_out in enumerate(expert_outputs):
                key = f"layers_{layer_idx}_expert_{i}"

                act = expert_out.to(torch.float32).detach().cpu()
                val = torch.norm(act, dim=(0, 1)).item() if act.dim() >= 2 else torch.norm(act).item()

                self.captured_expert_output[key] = [val]

        except Exception as e:
            print(f"[HOOK ERROR] Failed at layer {self.layer_count}: {e}")

    def cross_layer_expert_merge(self, model, frequency_list, group, layer_idx, layer_x,names):
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
                expert_up = model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_up[expert_idx]
                expert_down = model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_down[expert_idx]
                expert_gate = model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_gate[expert_idx]

                frequency = frequency_layer[expert_idx]  # Frequency of the current expert
                
                # Add weighted parameters to the accumulator
                if accumulated_up_proj_weight is None:
                    accumulated_up_proj_weight = expert_up * frequency
                    accumulated_down_proj_weight = expert_down * frequency
                    accumulated_gate_proj_weight = expert_gate * frequency
                else:
                    accumulated_up_proj_weight += expert_up * frequency
                    accumulated_down_proj_weight += expert_down * frequency
                    accumulated_gate_proj_weight += expert_gate * frequency
                
                # Accumulate total weight
                total_weight += frequency
    
        # Normalize by total weight
        if total_weight > 0:
            accumulated_up_proj_weight /= total_weight
            accumulated_down_proj_weight /= total_weight
            accumulated_gate_proj_weight /= total_weight
    
        # Set the merged parameters to the first expert of layer_idx
        # first_expert = model.model.layers[layer_idx].mlp.experts[0]
        first_name = names[layer_idx].split('_')
        first_layer = int(name[-1])
        first_encoder = False
        with torch.no_grad():
            model.model.layers[layer_idx].mlp.calculator.experts.weight_up[0]= nn.Parameter(accumulated_up_proj_weight.to(self.device))
            model.model.layers[layer_idx].mlp.calculator.experts.weight_down[0]= nn.Parameter(accumulated_down_proj_weight.to(self.device))
            model.model.layers[layer_idx].mlp.calculator.experts.weight_gate[0]= nn.Parameter(accumulated_gate_proj_weight.to(self.device))
    
        # Bind all experts in the range layer_idx to layer_x-1 to the first expert
        for cross_layer_idx in range(layer_idx, layer_x):
            name = names[cross_layer_idx].split('_')
            model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_up[0] = model.model.layers[layer_idx].mlp.calculator.experts.weight_up[0]
            model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_down[0] = model.model.layers[layer_idx].mlp.calculator.experts.weight_down[0]
            model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_gate[0] = model.model.layers[layer_idx].mlp.calculator.experts.weight_gate[0]
            for expert_idx in range(1,len(group[cross_layer_idx])):
                model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_up[expert_idx] = self.zero_param
                model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_down[expert_idx] = self.zero_param
                model.model.layers[cross_layer_idx].mlp.calculator.experts.weight_gate[expert_idx] = self.zero_param
            model.model.layers[cross_layer_idx].mlp.calculator.experts.remap_expert([0 for i in range(8)])  
        print(f"Cross-layer merge completed for layers {layer_idx} to {layer_x-1}")
        return model

    def _merge_experts_by_usage_frequency_weighting(
        self,
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
        # assert len(group) == len(frequency_list) == len(ffn.experts)
    
        # Convert group and frequency list to tensors for easier processing
        mapping = get_expert_remap_mapping(group)
        group_tensor = torch.tensor(group, dtype=torch.long)
        frequency_tensor = torch.tensor(frequency_list, dtype=torch.float)
    
        for label in group_tensor.unique():
            # Find the indices of experts in this group
            expert_indices = torch.where(group_tensor == label)[0]
            print(expert_indices)
            expert_indices_int = expert_indices.tolist()
            expert_indices_int = [int(i) for i in expert_indices_int]
            with torch.no_grad():
                # expert_names = ['expert_'+str(expert_idx) for expert_idx in expert_indices_int]
                # Accumulate weighted parameters for the group
                gate_weight_list = torch.stack(
                    [ffn.experts.weight_gate[i]* frequency_tensor[i] for i in expert_indices_int], dim=0
                )
                down_weight_list = torch.stack(
                    [ffn.experts.weight_down[i]* frequency_tensor[i] for i in expert_indices_int], dim=0
                )
                up_weight_list = torch.stack(
                    [ffn.experts.weight_up[i]* frequency_tensor[i] for i in expert_indices_int], dim=0
                )
    
                # Normalize the weights by their sum
                total_weight = torch.sum(frequency_tensor[expert_indices])
                gate_weight = torch.sum(gate_weight_list, dim=0) / (total_weight )
                down_weight = torch.sum(down_weight_list, dim=0) / (total_weight )
                up_weight = torch.sum(up_weight_list, dim=0) / (total_weight )
                
                # Set the merged weight to the first expert in the group
                ffn.experts.weight_gate[expert_indices_int[0]]= nn.Parameter(gate_weight.to(self.device))
                ffn.experts.weight_down[expert_indices_int[0]]= nn.Parameter(down_weight.to(self.device))
                ffn.experts.weight_up[expert_indices_int[0]]= nn.Parameter(up_weight.to(self.device))
    
                # Bind all experts in the group to the first expert (sharing parameters)
                for expert_idx in expert_indices_int:
                    if expert_idx == expert_indices_int[0]:
                        continue
                    ffn.experts.weight_gate[expert_idx] = self.zero_param
                    ffn.experts.weight_down[expert_idx] = self.zero_param
                    ffn.experts.weight_up[expert_idx] = self.zero_param
                print(expert_indices[0])

        ffn.experts.remap_expert(mapping)
        return ffn

    def adjust_groups_based_on_variance_similarity(self, frequency_list, group):
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
                if total_groups <= (self.num_expert * self.num_layer *self.args.percent):
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

    def get_updated_experts(self, usage_frequency_dict, group_dict, num_layers, names):
        result_dict = {}
        # result_dict['names'] = names
        # 遍历每一层
        layer_idx = 0
        group_idx = 0
        while layer_idx < num_layers:
            name = names[layer_idx]
            # 获取当前层的频率信息
            frequencies = usage_frequency_dict[layer_idx]
    
            # 获取当前层组别的唯一值，并将group中的所有元素转换为整数
            group_tensor = group_dict[layer_idx]
            print("The group tensor is")
            print(group_tensor)
            
            group_tensor = torch.tensor(group_tensor)  # Convert list to tensor
            group_tensor = group_tensor.int()  # 将所有元素转换为整数
            print(group_tensor)
            
            unique_groups = group_tensor.unique().tolist()
            print(unique_groups)
    
            # 创建一个字典，用于存储每个组的结果
            layer_result = {}
    
            if -2 not in group_tensor:  # 如果当前层的group不包含-2
                # 遍历所有组别
                for group in unique_groups:
                    # Ensure group_indices is always a list, even if it's a single index
                    group_indices = (group_tensor == group).nonzero(as_tuple=False).squeeze().tolist()
                    
                    # If group_indices is a single integer, convert it to a list
                    if isinstance(group_indices, int):
                        group_indices = [group_indices]  # Convert integer to list
                    
                    # Now, safely use the list comprehension
                    group_weights = [frequencies[idx] if idx in group_indices else 0 for idx in range(len(frequencies))]

                    # 计算该组的带权加和
                    weighted_sum = sum(group_weights)
    
                    # 归一化权重（除以带权加和）
                    if weighted_sum > 0:  # 防止除零错误
                        normalized_weights = [weight / weighted_sum for weight in group_weights]
                    else:
                        normalized_weights = group_weights  # 如果加和为零，保持原样（或可以处理为0）
    
                    # 将每一组的结果存入结果字典
                    result_dict[group_idx] = {
                        'layer': [layer_idx],
                        'indices': group_indices,
                        'weights': normalized_weights,
                        'weighted_sum': weighted_sum
                    }
                    group_idx += 1
    
                # 将该层的结果存入最终结果字典
                # result_dict[layer_idx] = layer_result
                layer_idx += 1  # 处理完当前层，继续下一个层
    
            else:  # 如果当前层的group包含-2
                # 跨层合并
                merged_group_indices = []
                merged_frequencies = []
                current_layer = layer_idx
                merged_layer_ids = []
                
                while current_layer < num_layers and -2 in group_dict[current_layer]:
                    group_tensor = group_dict[current_layer]
                    group_weights = usage_frequency_dict[current_layer]
                    merged_group_indices.extend(group_tensor)
                    merged_frequencies.extend(group_weights)
    
                    merged_layer_ids.append(current_layer)
                    current_layer += 1
    
                # 对合并的组进行带权加和计算
                total_weight = sum(merged_frequencies)
    
                if total_weight > 0:
                    merged_weights = [weight / total_weight for weight in merged_frequencies]
                else:
                    merged_weights = merged_frequencies  # 如果加和为零，保持原样（或可以处理为0）
    
                # 将合并后的结果存入字典
                merged_group = [-2]  # 假设-2代表跨层的组
                result_dict[group_idx] = {
                    'layer': merged_layer_ids,
                    'indices': merged_group,
                    'weights': merged_weights,
                    'weighted_sum': total_weight
                }
    
                # 存储结果字典
                # result_dict[layer_idx] = layer_result
    
                # 跳过所有跨层的层
                layer_idx = current_layer
                group_idx += 1
    
        return result_dict
    
    def assign_experts_to_groups_by_similarity(
        self,
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
        # print(num_experts)
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
            print(f"length of layer {layer_idx} is :{len(layer_expert_output)}")
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
        self,
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
        # 从 model 创建一个独立的新模型（保持结构）
        state_dict = model.state_dict()
        new_model = type(model)(model.config)
        new_model.load_state_dict(state_dict)
        # 使用频率信息来分配组
        group = self.assign_experts_to_groups_by_similarity(frequency_list, expert_output)
        group = self.adjust_groups_based_on_variance_similarity(frequency_list, group)
        self.X = self.get_updated_experts(frequency_list,group,self.num_layer,names)
        # print(group)
        # print("1")
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
                    new_model = self.cross_layer_expert_merge(new_model, frequency_list, group, layer_idx, layer_x,names)
                    # After merging, update layer_idx to layer_x (next unmerged layer)
                    layer_idx = layer_x
            else:
                # If no -2 in the group, proceed with the original logic
                name_layer = names[layer_idx].split('_')
                group_layer = group[layer_idx]  # Get the group for the current layer
                frequency_layer = frequency_list[layer_idx]  # Get frequency for the current layer
                print(f"Normal merging for layer {layer_idx}")
                new_model.model.layers[layer_idx].mlp.calculator = self._merge_experts_by_usage_frequency_weighting(
                    ffn=new_model.model.layers[layer_idx].mlp.calculator,
                    group=group_layer,
                    frequency_list=frequency_layer,
                )
                layer_idx += 1  # Move to the next layer
            print("done!")
        print("all done!")
        # model.to('cpu')
        torch.cuda.empty_cache()
        return new_model
        
    def save_model(self):
        """保存 LlamaMoEForCausalLM 中所有 expert_ 层的权重及其层名"""
        expert_weights = {}

        try:
            for group_idx, group_data in self.X.items():
                try:
                    layer_ids = group_data['layer']
                    group_indices = group_data['indices']

                    layer_min_idx = min(layer_ids)
                    layer_name = f'model.layers.{layer_min_idx}'

                    if isinstance(group_indices, int):
                        expert_min_idx = group_indices
                    elif isinstance(group_indices, (list, tuple)):
                        expert_min_idx = min(group_indices)
                    else:
                        raise TypeError(f"Unsupported type for group_indices: {type(group_indices)}")

                    if expert_min_idx < 0:
                        logger.warning(f"Expert index {expert_min_idx} is negative, resetting to 0.")
                        expert_min_idx = 0

                    # 构造 expert 的路径（基于 LlamaMoEDecoderLayer 中的 LinearGLUExperts）
                    base_path = f"{layer_name}.mlp.calculator.experts"

                    # 提取三个部分的参数 weight_gate / weight_up / weight_down
                    for param_type in ["weight_gate", "weight_up", "weight_down"]:
                        param_list = getattr(self.local_model._model.get_submodule(base_path), param_type, None)

                        if param_list is None or not isinstance(param_list, torch.nn.ParameterList):
                            raise AttributeError(f"Submodule {base_path} does not contain ParameterList '{param_type}'")

                        param_tensor = param_list[expert_min_idx].detach()  # 安全提取当前 expert 的权重
                        param_key = f"group_{group_idx}_{param_type}"
                        param_name = f"{base_path}.{param_type}[{expert_min_idx}]"

                        expert_weights[param_key] = {
                            'layer_name': param_name,
                            'weight': param_tensor
                        }

                except Exception as e:
                    logger.error(f"Error processing group {group_idx}: {e}", exc_info=True)
                    continue

            # 保存模型权重
            model_save_path = os.path.join(self.upload_folder, f"{self.ip}.pt")
            torch.save(expert_weights, model_save_path)
            logger.info(f"Model saved successfully at {model_save_path}")

            return model_save_path

        except Exception as e:
            logger.exception(f"Failed to save model: {e}")
            return None
        
    def download_files_from_all_nodes(self):
        """从所有节点下载文件"""
        # 读取当前的节点信息表TODO
        # with open(self.info_table_path, 'r') as f:
        #     node_info_table = json.load(f)

        # 遍历每个节点的IP
        for ip, node_info in self.node_info_table.items():
            model_path = node_info.get('model_path')  # 获取模型路径
            
            if model_path:
                # 尝试从该节点的 Flask 服务器下载文件
                try:
                    neighbor_port = 5000 + int(ip.split('.')[-1])
                    url = f'http://127.0.0.1:{neighbor_port}/download/{os.path.basename(model_path)}'
                    response = requests.get(url)

                    # 检查响应是否成功
                    if response.status_code == 200:
                        # 将下载的文件保存到本地
                        download_path = os.path.join(self.upload_folder, f"{ip}.pt")
                        with open(download_path, 'wb') as f:
                            f.write(response.content)

                        print(f"File downloaded: {download_path}")
                    else:
                        print(f"Failed to download file from {ip}: {response.text}")

                except requests.exceptions.RequestException as e:
                    print(f"Error downloading file from {ip}: {e}")

    def load_expert_weights(self, ip):
        """从对应的IP节点的文件中加载 expert 权重（支持 gate/up/down）"""
        file_path = os.path.join(self.upload_folder, f"{ip}.pt")
        if not os.path.exists(file_path):
            print(f"File for {ip} not found.")
            return None

        expert_weights = torch.load(file_path)
        gate_weights, up_weights, down_weights = [], [], []

        for key, value in expert_weights.items():
            if 'weight_gate' in key:
                gate_weights.append(value['weight'])
            elif 'weight_up' in key:
                up_weights.append(value['weight'])
            elif 'weight_down' in key:
                down_weights.append(value['weight'])

        return gate_weights, up_weights, down_weights


    def combine_all_weights_test(self):
        """从当前节点文件整合 gate/up/down 权重"""
        combined_gate, combined_up, combined_down = [], [], []
        gate_weights, up_weights, down_weights = self.load_expert_weights(self.ip)

        if gate_weights is not None:
            combined_gate.extend(gate_weights)
            combined_up.extend(up_weights)
            combined_down.extend(down_weights)

        return combined_gate, combined_up, combined_down


    def combine_all_weights(self):
        """从所有节点的文件中整合 gate/up/down 权重"""
        combined_gate, combined_up, combined_down = [], [], []

        for ip in self.update_solution.get('ip_node', []):
            gate_weights, up_weights, down_weights = self.load_expert_weights(ip)

            if gate_weights is not None:
                combined_gate.extend(gate_weights)
                combined_up.extend(up_weights)
                combined_down.extend(down_weights)
        if self.ip not in self.update_solution.get('ip_node', []):
            gate_weights, up_weights, down_weights = self.load_expert_weights(self.ip)
            if gate_weights is not None:
                combined_gate.extend(gate_weights)
                combined_up.extend(up_weights)
                combined_down.extend(down_weights)

        return combined_gate, combined_up, combined_down


    def upload_model(self, chunk_size_mb=50, max_retries=3):
        """分块上传模型文件，支持断点续传、进度条显示和自动重传"""
    
        model_save_path = self.save_model()
        # url = f"http://{self.ip}:{self.port}/upload"
        self.model_save = model_save_path
        print(f"Model saved locally at {model_save_path}")
        # file_size = os.path.getsize(model_save_path)
        # chunk_size = chunk_size_mb * 1024 * 1024  # 每块大小（默认50MB）
        # total_chunks = (file_size + chunk_size - 1) // chunk_size
    
        # # 计算文件sha256
        # file_hash = self.calculate_sha256(model_save_path)
    
        # session = requests.Session()
    
        # print(f"Uploading {file_size / (1024*1024):.2f}MB model to {url} in {total_chunks} chunks... (SHA256: {file_hash})")
    
        # with open(model_save_path, 'rb') as f:
        #     for chunk_index in tqdm(range(total_chunks), desc="Uploading Chunks", unit="chunk"):
        #         f.seek(chunk_index * chunk_size)
        #         chunk_data = f.read(chunk_size)
    
        #         success = False
        #         for attempt in range(max_retries):
        #             try:
        #                 files = {
        #                     'file': (f"{os.path.basename(model_save_path)}.part{chunk_index}", chunk_data)
        #                 }
        #                 data = {
        #                     'filename': os.path.basename(model_save_path),
        #                     'chunk_index': chunk_index,
        #                     'total_chunks': total_chunks,
        #                     'file_hash': file_hash
        #                 }
        #                 response = session.post(url, data=data, files=files, timeout=(10, 300))
    
        #                 if response.status_code == 200:
        #                     success = True
        #                     break
        #                 else:
        #                     print(f"[WARN] Chunk {chunk_index} upload failed with status {response.status_code}")
        #             except Exception as e:
        #                 print(f"[WARN] Exception while uploading chunk {chunk_index}: {str(e)}")
    
        #             time.sleep(2)  # 重试等待2秒
    
        #         if not success:
        #             print(f"[ERROR] Failed to upload chunk {chunk_index} after {max_retries} retries. Aborting...")
        #             return
    
        # print(f"Upload complete. Waiting for server to merge and verify {os.path.basename(model_save_path)}...")

    def send_node_info_to_neighbors(self):
        """将节点信息发送到所有邻居节点，增强版（带健康检查与异常保护）"""
        data = {
            'ip': self.ip,
            'hops': 0,
            'path': [self.ip],
            'timestamp': time.time(),
            'X': self.X,
            'Y': self.Y,
            'model_path': self.model_save,
        }
        global global_node_infor
        global_node_infor[self.ip] = {
                    'hops': data['hops'],
                    'path': data['path'],
                    'timestamp': data['timestamp'],
                    'X': data['X'],
                    'Y': data['Y'],
                    'model_path': data['model_path']
                }
        logger.info(f"Node info successfully sent")


    def forward_info_to_neighbors(self, data):
        """异步+重试转发接收到的节点信息到所有邻居节点"""
        def _send_to_neighbor(neighbor_ip, payload):
            neighbor_port = 5000 + int(neighbor_ip.split('.')[-1])
            neighbor_url = f"http://127.0.0.1:{neighbor_port}/receive_info"
    
            for attempt in range(3):
                try:
                    response = requests.post(neighbor_url, json=payload, timeout=5)
                    if response.status_code == 200:
                        print(f"[FORWARD] Success: Info forwarded to {neighbor_ip}")
                        return
                    else:
                        print(f"[FORWARD] Warning: Attempt {attempt+1} to {neighbor_ip} failed with status {response.status_code}")
                except Exception as e:
                    print(f"[FORWARD] Error: Attempt {attempt+1} to {neighbor_ip} raised exception: {e}")
                time.sleep(1)  # 小延迟重试
            print(f"[FORWARD] Failed: Giving up on forwarding to {neighbor_ip} after 3 attempts.")
    
        # 深拷贝 data 防止在主逻辑中被修改
        data = copy.deepcopy(data)
    
        # # 修改 hops/path 后再拷贝，确保是转发版本
        # data['hops'] += 1
        # data['path'].append(self.ip)
    
        # 启动异步转发线程
        def _forward_all():
            for neighbor in self.neighbors:
                if neighbor != data['ip']:
                    _send_to_neighbor(neighbor, data)
    
        threading.Thread(target=_forward_all, daemon=True).start()

    def should_update(self, model_fre, frequency_list):
        """
        判断 model_fre 和 frequency_list 之间的差值是否小于阈值 self.diff
        如果小于阈值，则返回 False，不更新
        否则返回 True，进行更新
        """
        # 计算 model_fre 和 frequency_list 之间的差值
        diff_value = np.linalg.norm(np.array(model_fre) - np.array(frequency_list))  # 使用欧几里得距离计算差值
        
        # 如果差值小于阈值，表示不需要更新
        if diff_value < self.diff:
            print(f"Skipping update: Difference {diff_value} is smaller than threshold {self.diff}")
            return False
        else:
            print(f"Update needed: Difference {diff_value} is greater than threshold {self.diff}")
            return True

    def run_inference_step(self):
        """只进行推理 + 捕获 + 分发，不执行更新"""
        print(f"[{self.ip}] Inference Step Starting")
        selected_task = random.choice(self.expanded_task_list)
        self.start_time = datetime.now()
        self.selected_task = selected_task
        self.captured_outputs = {}
        self.captured_expert_output = {}

        self.hook_handles = []
        self.register_hooks(self.local_model)
        print(f'selected task is: {selected_task}')
        results = lm_eval.simple_evaluate(
            model=self.local_model,
            tasks=[self.selected_task],
            num_fewshot=0,
            task_manager=self.task_manager,
            batch_size=5,
            limit=100,
            device=self.device,
        )
        print(f"[{self.ip}] Inference Results (Before Update): {results['results']}")

        frequency_list, names_fre = map_to_range(self.captured_outputs, [])
        expert_output, names_out = map_to_range(self.captured_expert_output, [])

        self.being_update = True
        self.being_Information = True
        group = self.assign_experts_to_groups_by_similarity(frequency_list, expert_output)
        group = self.adjust_groups_based_on_variance_similarity(frequency_list, group)
        self.Y = self.get_updated_experts(frequency_list, group, self.num_layer, names_fre)
        self.upload_model()
        self.send_node_info_to_neighbors()



    def run_update_step(self, timeout=100):
        """仅执行更新步骤，依赖于其他节点已发送的消息"""
        # 检查是否可线性更新
        self.update_ava, self.update_solution = self.check_linear_valuable()
        print(f'The update_ava is: {self.update_ava} and the updated solution is: {self.update_solution}')
        if self.update_ava:
            self.being_Information = False
            print(f"[QUEUE] Update solution ready, node will update.")

        if self.being_Information:
            print(f"[{self.ip}] Update timeout. No solution received.")
            self.being_update = False
            return

        print(f"[{self.ip}] Update info received. Applying update.")
        self.update_process()



    def model_inference_and_update(self):
        """模拟模型推理并触发更新操作"""
        # 模拟推理过程
        print(f"Node {self.ip} performing model inference...")
        tasks = self.expanded_task_list
        if tasks:
            self.start_time = datetime.now()
            # 选择一个任务
            selected_task = random.choice(tasks)
            self.captured_outputs = {}
            self.captured_expert_output = {}
            self.selected_task = selected_task
            self.hook_handles = []
            self.register_hooks(self.local_model)
            results = lm_eval.simple_evaluate( # call simple_evaluate squad_completion
                model=self.local_model,
                tasks=[selected_task],
                num_fewshot=0,
                task_manager=self.task_manager,
                batch_size = 1,
                limit = 200,#500 TODO
                device = self.device,
            )
            print(f'Before Update in time {datetime.now()}: {results['results']}')
            
            frequency_list,names_fre = map_to_range(self.captured_outputs,[])
            expert_output,names_out = map_to_range(self.captured_expert_output,[])
            if np.linalg.norm(np.array(self.model_fre) - np.array(frequency_list)) > self.diff:
                self.being_update = True
                self.being_Information = True
                group = self.assign_experts_to_groups_by_similarity(frequency_list, expert_output)
                group = self.adjust_groups_based_on_variance_similarity(frequency_list, group)
                # print("This is the group like")
                # print(group)
                self.Y = self.get_updated_experts(frequency_list,group,self.num_layer,names_fre)
                # print("This is Y looks like:")
                # print(self.Y)
                # self.adjust_groups_based_on_variance_similarity(frequency_list, group)
                
            # 模拟更新过程：上传模型文件和发送节点信息
            self.upload_model()
            print(f'DEBUG in {self.ip}: Start to send node info: ')
            self.send_node_info_to_neighbors()
#######################
            while self.being_update:
                self.update_process()
            results = lm_eval.simple_evaluate( # call simple_evaluate squad_completion
                model=self.local_model,
                tasks=[selected_task],
                num_fewshot=0,
                task_manager=self.task_manager,
                batch_size = 1,
                limit = 200,#500 TODO
                device = self.device,
            )
            print(f'After Update in time {datetime.now()}: {results['results']}')
            
        # 设置定时任务，每隔一定时间触发更新操作
        # threading.Timer(10, self.model_inference_and_update).start()
    def model_inference_and_update_loop(self):
        while True:
            try:
                self.model_inference_and_update()
            except Exception as e:
                print(f"Inference loop error: {e}")
            time.sleep(10)

    def update_process(self):
        # delta_time = datetime.now() - self.start_time
        # while self.being_Information and delta_time.total_seconds() < 3600:
        #     time.sleep(1)
        #     delta_time = datetime.now() - self.start_time
        # # 处理接收到的信息 - 没有解
        # if self.being_Information:
        #     # 没得到更新
        #     print(f"Node {self.ip} cannot get the solution.")
        #     self.being_update = False
        #     return 
        # # 处理接收到的信息 - 有解 - 开始更新
        print(f"Node {self.ip} got the solution.")
        combined_gate, combined_up, combined_down = self.combine_all_weights()
        combined_gate = [x.to(self.device) for x in combined_gate]
        combined_up = [x.to(self.device) for x in combined_up]
        combined_down = [x.to(self.device) for x in combined_down]
        solution = self.update_solution.get('solution', [])
        # 遍历 solution 中的每个 ip_node 对应的权重
        for ip, weights in enumerate(solution):
            ip_weights = weights
            print(f"Example weights: {solution[0]}")
            print(f"Type of weights: {type(solution[0])}")
            print(f"Len of combined_gate: {len(combined_gate)}")
            weighted_gate = torch.zeros_like(combined_gate[0], device=self.device)
            weighted_up = torch.zeros_like(combined_up[0], device=self.device)
            weighted_down = torch.zeros_like(combined_down[0], device=self.device)

            print(f"len(ip_weights): {len(ip_weights)}")
            print(f"len(combined_gate): {len(combined_gate)}")

            for i, weight in enumerate(ip_weights):
                weighted_gate += weight * combined_gate[i]
                weighted_up += weight * combined_up[i]
                weighted_down += weight * combined_down[i]

            group_data = self.Y[ip]
            layer_ids = group_data['layer']
            group_indices = group_data['indices']
            normalized_weights = group_data['weights']

            min_layer_idx = min(layer_ids)
            layer_name = f'model.layers.{min_layer_idx}'
            min_expert_idx = max(min(group_indices), 0)

            base_path = f"{layer_name}.mlp.calculator.experts"
            experts_module = self.local_model._model.get_submodule(base_path)

            with torch.no_grad():
                experts_module.weight_gate[min_expert_idx] = nn.Parameter(weighted_gate.to(self.device))
                experts_module.weight_up[min_expert_idx]= nn.Parameter(weighted_up.to(self.device))
                experts_module.weight_down[min_expert_idx]= nn.Parameter(weighted_down.to(self.device))

                if len(layer_ids) == 1:#TODO
                    for other_expert in group_indices:
                        if other_expert != min_expert_idx:
                            experts_module.weight_gate[other_expert] = self.zero_param
                            experts_module.weight_up[other_expert] = self.zero_param
                            experts_module.weight_down[other_expert] = self.zero_param
                            experts_module.remap_expert_idx(other_expert, min_expert_idx)
                else:
                    for other_layer_idx in layer_ids:
                        other_layer_name = f'model.layers.{other_layer_idx}'
                        # if other_layer_name == layer_name:
                        #     continue
                        other_module = self.local_model._model.get_submodule(f"{other_layer_name}.mlp.calculator.experts")
                        for expert_idx in range(self.num_expert):
                            if expert_idx != min_expert_idx:
                                other_module.weight_gate[expert_idx] = self.zero_param
                                other_module.weight_up[expert_idx] = self.zero_param
                                other_module.weight_down[expert_idx] = self.zero_param
                            else:
                                other_module.weight_gate[expert_idx] = experts_module.weight_gate[min_expert_idx]
                                other_module.weight_up[expert_idx] = experts_module.weight_up[min_expert_idx]
                                other_module.weight_down[expert_idx] = experts_module.weight_down[min_expert_idx]
                            other_module.remap_expert_idx(expert_idx, min_expert_idx)
        self.update_solution = {}
        self.being_update = False
        self.start_time = datetime.now()

        
####################################
    def start_flask_server(self):
        """启动Flask服务器（非阻塞线程 + 动态端口）"""
        print(f"Node {self.ip} Flask server is running on port {self.port}...")
        threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False),
            daemon=True
        ).start()

    def start_inference_thread(self):
        """此函数可选，如果你仍希望类内启动单个推理进程"""
        from multiprocessing import Process
        p = Process(target=self.model_inference_and_update_loop)
        p.start()

    def stop_flask_server(self):
        """停止Flask服务器"""
        func = self.app.get_send_file_max_age_func()
        func()


def build_edge_network(num_nodes, neighbors_count, expanded_task_list, base_info_path='node_info.json', num_gpus=8):
    # 随机生成唯一的IP地址
    def generate_unique_ip(existing_ips):
        while True:
            ip = f"{random.randint(1, 255)}"
            if ip not in existing_ips:
                existing_ips.add(ip)
                return ip

    # 提取任务组和任务字典
    task_group = expanded_task_list
    # print(task_group)
    # print(task_dict)
    # 创建所有节点
    nodes = []
    ip_list = []
    existing_ips = set()

    # 使用轮循的方式分配 GPU
    gpu_id_list = list(range(num_gpus)) * (num_nodes // num_gpus) + list(range(num_nodes % num_gpus))  # 轮流分配 GPU

    # 存储已经分配的任务，确保任务不重复
    assigned_tasks = set()

    for _ in range(num_nodes):
        ip = generate_unique_ip(existing_ips)
        ip_list.append(ip)

    # 为每个节点随机选择邻居
    node_neighbors = {}
    for ip in ip_list:
        neighbors = random.sample([x for x in ip_list if x != ip], neighbors_count)
        node_neighbors[ip] = neighbors

    # 确保邻居关系是双向的
    for ip, neighbors in node_neighbors.items():
        for neighbor in neighbors:
            if ip not in node_neighbors[neighbor]:
                node_neighbors[neighbor].append(ip)

    # 保存每个节点的网络架构到独立文件
    for ip, neighbors in node_neighbors.items():
        # 每个节点对应一个独立的文件，确保文件名唯一
        network_info_path = f"network_info/{ip}_network_info.json"
        network_info = {ip: neighbors}
        
        with open(network_info_path, 'w') as f:
            json.dump(network_info, f)
            
    # # 随机抽取 num_nodes 个任务 TODO
    # if len(expanded_task_list) < num_nodes:
    #     raise ValueError("expanded_task_list长度不足，无法为所有节点分配唯一任务")

    selected_tasks = random.choices(expanded_task_list, k=num_nodes) #TODO
    
    # 创建节点实例，分配 GPU 和任务
    nodes = []
    for idx, (ip, neighbors) in enumerate(node_neighbors.items()):
        gpu_id = gpu_id_list[idx]
        selected_task = selected_tasks[idx] #TODO: selected_tasks[idx]expanded_task_list[0]

        node = Node(
            ip=ip,
            args=args,
            task=selected_task,
            neighbors=neighbors,
            info_table_path=f"network_info/{ip}_info.json",
            gpu_id=gpu_id
        )
        nodes.append(node)

    return nodes, node_neighbors

def initialize_model_in_parallel(nodes):
    """
    以并行的方式初始化节点的模型，确保每组内部只有一个节点在同一时间进行初始化。
    """
    # 线程锁，用于确保每组内只有一个节点在同一时间进行初始化
    group_locks = {group_id: threading.Lock() for group_id in range(6)}

    def initialize_node(node):
        with group_locks[node.gpu_id]:  # 确保同一组内节点是顺序进行初始化
            node.initialize_model()

    # 使用多线程并行初始化节点
    threads = []
    for node in nodes:
        thread = threading.Thread(target=initialize_node, args=(node,))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成初始化
    for thread in threads:
        thread.join()
        
def initialize_model_sequentially(nodes):
    """
    顺序初始化模型，统一并发启动 Flask，并确认上线后，再统一启动推理。
    """
    # 【1】阶段1：顺序初始化模型
    for node in nodes:
        node.initialize_model()

    # # 【2】阶段2：启动所有节点的 Flask Server（并行启动）
    # threads = []
    # for node in nodes:
    #     thread = threading.Thread(target=node.start_flask_server)
    #     thread.start()
    #     threads.append(thread)

    # Step 1: 所有节点先推理 + 广播信息
    for node in nodes:
        node.run_inference_step()
    print('Start to Update Step!!')
    # Step 2: 所有节点再进行更新（如果需要）
    for node in nodes:
        node.run_update_step(timeout=100)


# 示例使用
num_nodes = 3
neighbors_count = 1
base_info_path = 'node_info.json'
expanded_task_list = [
            # 'AraDiCE_ArabicMMLU_high_humanities_history_egy',
            # 'AraDiCE_ArabicMMLU_high_humanities_islamic-studies_lev',
            # 'AraDiCE_piqa_egy',
            # 'AraDiCE_ArabicMMLU_high_stem_biology_egy',
            # 'arc_easy',
            # # 'arc_challenge',
            # # 'anagrams1',
            # 'anli_r2',
            # 'anli_r1',
            # 'arabic_leaderboard_arabic_mmlu_high_school_statistics_light',
            'coqa',
            'eq_bench',
            # 'fda',
            'cola',
            'mnli',
            'mrpc',
            'qnli',
            # 'qqp',
            'rte',
            # 'sst',
            'wnli',
            # # 'gpqa_main_zeroshot',
            # # 'gpqa_diamond_zeroshot',
            # # 'gpqa_extended_zeroshot',
            # # 'gpqa_main_n_shot',
            # # 'gpqa_diamond_n_shot',
            # # 'gpqa_extended_n_shot',
            # # 'gpqa_main_generative_n_shot',
            # # 'gpqa_diamond_generative_n_shot',
            # # 'gpqa_extended_generative_n_shot',
            # # 'gpqa_main_cot_zeroshot',
            # # 'gpqa_diamond_cot_zeroshot',
            # # 'gpqa_extended_cot_zeroshot',
            # # 'gpqa_main_cot_n_shot',
            # # 'gpqa_diamond_cot_n_shot',
            # # 'gpqa_extended_cot_n_shot',
            # # 'lambada_openai',
            # # 'lambada_standard',
            # # 'leaderboard_bbh_causal_judgement',
            # # 'leaderboard_bbh_disambiguation_qa',
            # # 'leaderboard_bbh_hyperbaton',
            # # # 'leaderboard_bbh_logical_deduction_five_objects',
            # # 'leaderboard_bbh_navigate',
            # # # 'leaderboard_bbh_object_counting',
            # # 'leaderboard_bbh_reasoning_about_colored_objects',
            # # 'leaderboard_bbh_ruin_names',
            # # # 'leaderboard_bbh_salient_translation_error_detection',
            # # # 'leaderboard_bbh_sports_understanding',
            # 'leaderboard_bbh_temporal_sequences',
            # # 'leaderboard_bbh_tracking_shuffled_objects_seven_objects',
            # # 'leaderboard_bbh_tracking_shuffled_objects_three_objects',
            # 'leaderboard_bbh_web_of_lies', 
            'mastermind_24_easy',
            # 'mastermind_24_hard',
            'mastermind_35_easy',
            # 'mastermind_35_hard',
            'mastermind_46_easy',
            # 'mastermind_46_hard',
            'logiqa',
            # # 'mmlu',
            # 'mmlu_stem',
            # 'mmlu_humanities',
            # 'mmlu_other',
            # # 'mmlu_pro_psychology',
            # # 'mmlu_pro_physics',
            # # 'mmlu_pro_philosophy',
            # # 'mmlu_pro_other',
            # # 'mmlu_pro_math',
            # # 'mmlu_pro_law',
            # # 'mmlu_pro_history',
            # # 'mmlu_pro_health',
            # # 'mmlu_pro_engineering',
            # # 'mmlu_pro_economics',
            # # 'mmlu_pro_computer_science',
            # # 'mmlu_pro_chemistry',
            # 'mmlu_social_sciences',
            'openbookqa',
            'piqa',
            'sciq',
            'boolq',
            'cb',
            'copa',
            'multirc',
            # 'record',
            'rte',
            'wic',
            'wsc',
            # 'super_glue-boolq-t5-prompt',
            # 'super_glue-cb-t5-prompt',
            # 'super_glue-copa-t5-prompt',
            # 'super_glue-multirc-t5-prompt',
            # 'super_glue-record-t5-prompt',
            # 'super_glue-rte-t5-prompt',
            # 'super_glue-wic-t5-prompt',
            # 'super_glue-wsc-t5-prompt',
            # 'truthfulqa_mc1',
            # 'truthfulqa_mc2',
            # 'truthfulqa_gen',
            'winogrande'
            ]
nodes, network_info = build_edge_network(num_nodes, neighbors_count, expanded_task_list, base_info_path)
print(f"网络架构：{network_info}")

# 启动节点初始化
initialize_model_sequentially(nodes)

# # 启动推理
# for node in nodes:
#     node.start_inference()

try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("Exiting gracefully.")
