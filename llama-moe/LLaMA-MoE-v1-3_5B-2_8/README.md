---
license: apache-2.0
language:
- en
tags:
- MoE
---
# LLaMA-MoE-v1-3.5B (2/8)

[[💻 Code]](https://github.com/pjlab-sys4nlp/llama-moe) | [[📜 Technical Report]](https://github.com/pjlab-sys4nlp/llama-moe/blob/main/docs/LLaMA_MoE.pdf)

👋 Very nice to meet you here~

❤️ This repo contains the model `LLaMA-MoE-v1-3.5B (2/8)`, which activates 2 out of 8 experts (3.5B parameters).
This model is NOT fine-tuned by instruction pairs, so it may not be good enough to act like a chatbot.

📢 LLaMA-MoE is a series of Mixture-of-Expert (MoE) models based on [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b-hf).
You can find the code for training this model at [this repo](https://github.com/pjlab-sys4nlp/llama-moe).

💎 This series of models are obtained by partitioning original LLaMA FFNs into experts and further continual pre-training.
The total model size is only 6.7B parameters, which is very convenient for deployment and research usage.
More details could be found at [our technical report](https://arxiv.org/).

## 🚀 QuickStart

```python
# python>=3.10

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:0")

input_text = "Suzhou is famous of"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")

pred = model.generate(**inputs, max_length=50, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three
```

## 📊 Performance

| Model                     | \#Activated Experts | \#Experts | \#Activated Params |                                   Links                                   |
| :------------------------ | :-----------------: | :-------: | :----------------: | :-----------------------------------------------------------------------: |
| **LLaMA-MoE-3.0B**        |          2          |    16     |        3.0B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_0B-2_16) |
| **LLaMA-MoE-3.5B (4/16)** |          4          |    16     |        3.5B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16) |
| **LLaMA-MoE-3.5B (2/8)**  |          2          |     8     |        3.5B        | [[🤗 HF Weights]](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8)  |


| Model                                                                                 |   SciQ   |   PIQA   | WinoGrande |  ARC-e   | ARC-c (25) | HellaSwag (10) |  LogiQA  | BoolQ (32) | LAMBADA  | NQ (32)  |  MMLU (5) | Average |
| :------------------------------------------------------------------------------------ | :------: | :------: | :--------: | :------: | :--------: | :------------: | :------: | :--------: | :------: | :------: | :-------: | :-----: |
| [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)                                  |   78.9   |   74.8   |    60.8    |   54.4   |    34.0    |      61.4      |   25.8   |    63.3    |   63.6   |   10.7   |   25.8    |  50.3   |
| [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b)                          |   83.2   |   73.6   |    59.6    |   58.8   |    36.7    |      60.7      |   28.1   |    65.9    |   64.6   |   8.7    |   26.8    |  51.5   |
| [INCITE-BASE-3B](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) |   85.6   |   73.9   |    63.5    |   61.7   |    40.3    |      64.7      |   27.5   |    65.8    |   65.4   |   15.2   |   27.2    |  53.7   |
| [Open-LLaMA-3B-v2](https://huggingface.co/openlm-research/open_llama_3b_v2)           |   88.0   |   77.9   |    63.1    |   63.3   |    40.1    |      71.4      |   28.1   |    69.2    |   67.4   |   16.0   |   26.8    |  55.6   |
| [Sheared-LLaMA-2.7B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B)         |   87.5   |   76.9   |    65.0    |   63.3   |    41.6    |      71.0      |   28.3   |    73.6    |   68.3   |   17.6   | **27.3**  |  56.4   |
| **LLaMA-MoE-3.0B**                                                                    |   84.2   |   77.5   |    63.6    |   60.2   |    40.9    |      70.8      | **30.6** |    71.9    |   66.6   |   17.0   |   26.8    |  55.5   |
| **LLaMA-MoE-3.5B (4/16)**                                                             |   87.6   | **77.9** |    65.5    | **65.6** |  **44.2**  |    **73.3**    |   29.7   |  **75.0**  | **69.5** | **20.3** |   26.8    |  57.7   |
| **LLaMA-MoE-3.5B (2/8)**                                                              | **88.4** |   77.6   |  **66.7**  |   65.3   |    43.1    |    **73.3**    |   29.6   |    73.9    |   69.4   |   19.8   |   27.0    |  57.6   |

## 📖 Details

Training Data: 200B tokens from [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) with the same data sampling weights as [Sheared LLaMA](https://arxiv.org/abs/2310.06694).

## 📃 Citation

```bibtex
@article{llama-moe,
  title={LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training},
  author={Tong Zhu and Xiaoye Qu and Daize Dong and Jiacheng Ruan and Jingqi Tong and Conghui He and Yu Cheng},
  journal={arXiv preprint arXiv:2406.16554},
  year={2024},
  url={https://arxiv.org/abs/2406.16554},
}
```