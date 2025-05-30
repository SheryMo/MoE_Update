# python>=3.10

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:0")
with open("model_structure_llama0.txt", "w") as f:
    # f.write("Model Architecture:\n\n")
    # f.write(str(model))
    # f.write("\n\nModel Layers:\n\n")
    expert_up = model.model.layers[0].mlp.calculator.experts.weight_up[0]
    f.write(str(expert_up))
    f.write(expert_up.weight)
