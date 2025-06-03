import json, os
import ast

tasks_list = {
    "P_tasks": ['Connected','Diameter','Distance','Neighbor'],
    "NP_tasks": ['GED','TSP','MCP','MCS','MIS','MVC']
}

def get_int(ans):
    try:
        ans = ast.literal_eval(ans)
        if type(ans) == int:
            return ans
        if type(ans) == list and len(ans) == 1 and type(ans[0]) == int:
            return ans[0]
        return -1
    except:
        return -1

final_result = {}
for path in [
    "original_logs/Llama-3.2-3B-Instruct_eval_result.json",
    "original_logs/Qwen2.5-3B-Instruct_eval_result.json",
    "original_logs/G1-3B_eval_result.json",
    "original_logs/LLaMA2-7B-RFT_eval_result.json",
    "original_logs/LLaMA2-7B-DPO_eval_result.json",
    "original_logs/Llama-3.1-8B-Instruct_eval_result.json",
    "original_logs/DeepSeek-R1-Distill-Qwen-7B_eval_result.json",
    "original_logs/Qwen2.5-7B-Instruct_eval_result.json",
    "original_logs/G1-7B_eval_result.json"
]:
    results = json.load(open(path))
    meta_data = results["meta_data"]

    for fix_task in ["Diameter_easy", "Diameter_hard"]:
        meta_data[fix_task]['acc'] = 0
        for item in results[fix_task]:
            ## There are cases in which the models fail to follow the answer format precisely.
            ## We specially handle one such case where the models output the numerical answers directly instead of providing the list of nodes/edges as expected.
            ## Excluding this will not invalidate our claims.
            meta_data[fix_task]['acc'] += (item['correct']) or (int(item["golden_answer"] == get_int(item["generated_answer"])))
        meta_data[fix_task]['acc'] /= meta_data[fix_task]['total']

    summary = {}
    for diff in ['easy', 'hard']:
        for tasks_name, tasks in tasks_list.items():
            total, acc = 0, 0
            for task in tasks:
                total += meta_data[f'{task}_{diff}']['total']
                acc += meta_data[f'{task}_{diff}']['acc'] * meta_data[f'{task}_{diff}']['total']
                summary[f'{task}_{diff}'] = meta_data[f'{task}_{diff}']['acc'] * meta_data[f'{task}_{diff}']['total']
            summary[f'{tasks_name}_{diff}'] = 100 * acc / total

    from numpy import mean
    summary['avg'] = 0.2 * summary['P_tasks_easy'] + 0.2 * summary['P_tasks_hard'] + 0.3 * summary['NP_tasks_easy'] + 0.3 * summary['NP_tasks_hard']
            
    final_result[path] = summary

for path in [
    "original_logs/Llama-3.2-3B-Instruct_eval_result.json",
    "original_logs/Qwen2.5-3B-Instruct_eval_result.json",
    "original_logs/G1-3B_eval_result.json",
    "original_logs/LLaMA2-7B-RFT_eval_result.json",
    "original_logs/LLaMA2-7B-DPO_eval_result.json",
    "original_logs/Llama-3.1-8B-Instruct_eval_result.json",
    "original_logs/DeepSeek-R1-Distill-Qwen-7B_eval_result.json",
    "original_logs/Qwen2.5-7B-Instruct_eval_result.json",
    "original_logs/G1-7B_eval_result.json"
    ]:
    print(f'{path}', end=' ')
    print(f'& {final_result[path]["avg"]:.2f}', end=' ')
    print(' \\\\ ')
print('')

for diff in ['easy', 'hard']:
    for path in [
        "original_logs/Llama-3.2-3B-Instruct_eval_result.json",
        "original_logs/Qwen2.5-3B-Instruct_eval_result.json",
        "original_logs/G1-3B_eval_result.json",
        "original_logs/LLaMA2-7B-RFT_eval_result.json",
        "original_logs/LLaMA2-7B-DPO_eval_result.json",
        "original_logs/Llama-3.1-8B-Instruct_eval_result.json",
        "original_logs/DeepSeek-R1-Distill-Qwen-7B_eval_result.json",
        "original_logs/Qwen2.5-7B-Instruct_eval_result.json",
        "original_logs/G1-7B_eval_result.json"
        ]:
        print(f'{path}', end=' ')
        cur_result = final_result[path]
        for tasks_name, tasks in tasks_list.items():
            for task in tasks:
                print(f'& {cur_result[f"{task}_{diff}"]:.2f}', end=' ')
                
        print(' \\\\ ')
    print('')