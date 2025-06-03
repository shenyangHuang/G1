import json
import os
import csv
from collections import defaultdict


model_list = [
    "Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "CoT-SFT-3B", 
    "G1-3B", "Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-7B-Instruct", "CoT-SFT-7B", "G1-7B",
]


def cal_avg_acc(data, task_list):
    avg_acc = 0
    for task in task_list:
        avg_acc += data["meta_data"][task]["acc"]
    avg_acc /= len(task_list)
    return avg_acc


def print_acc(data_path):
    with open(data_path + "_node_classification.json", 'r') as f:
        data = json.load(f)
    node_cora_acc = cal_avg_acc(data, ["cora_with_label", "cora_without_label"])
    node_pubmed_acc = cal_avg_acc(data, ["pubmed_with_label", "pubmed_without_label"])
    
    with open(data_path + "_link_prediction.json", 'r') as f:
        data = json.load(f)
    link_cora_acc = cal_avg_acc(data, ["cora_with_title", "cora_without_title"])
    link_pubmed_acc = cal_avg_acc(data, ["pubmed_with_title", "pubmed_without_title"])

    acc = (node_cora_acc + node_pubmed_acc + link_cora_acc + link_pubmed_acc) / 4
    
    print(f"model: {data_path.replace('.json', '')}")
    print(f"Node-Cora: {node_cora_acc * 100: .2f}")
    print(f"Node-PubMed: {node_pubmed_acc * 100: .2f}")
    print(f"Link-Cora: {link_cora_acc * 100: .2f}")
    print(f"Link-PubMed: {link_pubmed_acc * 100: .2f}")
    print(f"Acc: {acc * 100: .2f}")
    print("\n")


for model in model_list:
    print_acc(model)