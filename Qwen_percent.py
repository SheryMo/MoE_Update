import subprocess

# 定义需要运行的 dataset 列表
datasets = ["winogrande", "my_custom_dataset", "another_dataset"]

# 遍历并运行命令
for dataset in datasets:
    command = f"python script.py --dataset {dataset}"
    print(f"Running: {command}")
    # 执行命令
    subprocess.run(command, shell=True)
