import json
import os
import csv
from collections import defaultdict


linear_list = ['connectivity', 'cycle', 'bipartite', 'topology']
poly_list = ['shortest', 'triangle', 'flow']
np_list = ['hamilton', 'substructure']

model_list = [
    "Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "G1-3B", 
    "Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Qwen-7B",
    "GraphWiz-LLaMA2-7B-RFT", "GraphWiz-LLaMA2-7B-DPO", 
    "Qwen2.5-7B-Instruct", "G1-7B",
]


def cal_avg_acc(data, task_list):
    avg_acc = 0
    for task in task_list:
        avg_acc += data["meta_data"][task]["acc"]
    avg_acc /= len(task_list)
    return avg_acc


def print_acc(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    acc = data["meta_data"]["average"]["acc"]

    linear_avg_acc = cal_avg_acc(data, linear_list)
    poly_avg_acc = cal_avg_acc(data, poly_list)
    np_avg_acc = cal_avg_acc(data, np_list)

    print(f"model: {data_path.replace('.json', '')}")
    print(f"Linear: {linear_avg_acc * 100: .2f}")
    print(f"Polynormial: {poly_avg_acc * 100: .2f}")
    print(f"NP-Complete: {np_avg_acc * 100: .2f}")
    print(f"Average: {acc * 100: .2f}")
    print("\n")


for model in model_list:
    eval_file = model + '.json'
    if not os.path.exists(eval_file):
        print(f"Evaluation file {eval_file} does not exist.")
        continue
    print_acc(eval_file)