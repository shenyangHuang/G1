import json
import os
import csv
from collections import defaultdict


easy_list = ['node_number', 'dominating_set', 'common_neighbor', 'edge_number', 'neighbor', 'bfs', 'has_cycle', 'dfs', 'minimum_spanning_tree', 'edge_existence', 'is_regular', 'degree', 'is_tournament', 'density']
medium_list = ['adamic_adar_index', 'clustering_coefficient', 'connected_component_number', 'bipartite_maximum_matching', 'local_connectivity', 'jaccard_coefficient', 'min_edge_covering', 'is_eularian', 'degree_centrality', 'is_bipartite', 'resource_allocation_index']
hard_list = ['max_weight_matching', 'closeness_centrality', 'traveling_salesman_problem', 'strongly_connected_number', 'shortest_path', 'center', 'diameter', 'barycenter', 'radius', 'topological_sort', 'periphery', 'betweenness_centrality', 'triangles', 'avg_neighbor_degree', 'harmonic_centrality', 'bridges']
challenging_list = ['isomophic_mapping', 'global_efficiency', 'maximal_independent_set', 'maximum_flow', 'wiener_index', 'hamiltonian_path', 'min_vertex_cover']


model_list = [
    "GPT4o-mini", "o3-mini", "Llama-3.2-3B-Instruct", "Qwen2.5-3B-Instruct", "Direct-SFT-3B", "CoT-SFT-3B",
    "G1-3B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-Math-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B", "GraphWiz-LLaMA2-7B-RFT", "GraphWiz-LLaMA2-7B-DPO",
    "Direct-SFT-7B", "CoT-SFT-7B", "G1-7B", "Llama-3.1-70B-Instruct", "Qwen2.5-72B-Instruct",
    "G1-Zero-3B", "G1-Zero-7B"
]


def write_to_csv():
    acc_dict = defaultdict(list)
    for model in model_list:
        eval_file = model + '_eval_result.json'
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        for task in easy_list:
            acc_dict[model].append(f'{data["meta_data"][task]["acc"]*100:.2f}')
        for task in medium_list:
            acc_dict[model].append(f'{data["meta_data"][task]["acc"]*100:.2f}')
        for task in hard_list:
            acc_dict[model].append(f'{data["meta_data"][task]["acc"]*100:.2f}')
        for task in challenging_list:
            acc_dict[model].append(f'{data["meta_data"][task]["acc"]*100:.2f}')
        
    with open("accuracy.csv", "w") as f:
        writer = csv.writer(f)
       
        writer.writerow(["Model"] + easy_list + medium_list + hard_list + challenging_list)
        for model, accs in acc_dict.items():
            writer.writerow([model] + accs)



def cal_avg_acc(data, task_list):
    avg_acc = 0
    for task in task_list:
        avg_acc += data["meta_data"][task]["acc"]
    avg_acc /= len(task_list)
    return avg_acc


def print_acc(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    if "average" in data["meta_data"].keys():
        acc = data["meta_data"]["average"]["acc"]
    else:
        acc = data["meta_data"]["Graph_acc"]["acc"]
    easy_avg_acc = cal_avg_acc(data, easy_list)
    medium_avg_acc = cal_avg_acc(data, medium_list)
    hard_avg_acc = cal_avg_acc(data, hard_list)
    challenging_avg_acc = cal_avg_acc(data, challenging_list)

    print(f"model: {data_path.replace('.json', '')}")
    print(f"Easy: {easy_avg_acc * 100: .2f}")
    print(f"Medium: {medium_avg_acc * 100: .2f}")
    print(f"Hard: {hard_avg_acc * 100: .2f}")
    print(f"Challenging: {challenging_avg_acc * 100: .2f}")
    print(f"Average: {acc * 100: .2f}")
    print("\n")


for model in model_list:
    eval_file = model + '.json'
    if not os.path.exists(eval_file):
        print(f"Evaluation file {eval_file} does not exist.")
        continue
    print_acc(eval_file)