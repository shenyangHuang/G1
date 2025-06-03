from transformers import AutoTokenizer
import json
import os
import argparse
import re
import ast
import vllm
import argparse
import datasets
import networkx as nx

from string_utils import strip_string
from math_utils import remove_boxed, last_boxed_only_string


QUERY_TEMPLATE = """
{instruction}

Solve the above problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.
""".strip()


def extract_solution(solution_str, model_name):
    if 'graphwiz' in model_name.lower():
        last_index = solution_str.rfind("###")
        generated_answer = solution_str[last_index + 3:].replace(".", "").strip() if last_index != -1 else None
    else:
        possible_ans = last_boxed_only_string(solution_str)
        generated_answer = strip_string(remove_boxed(possible_ans)) if possible_ans else None
    return generated_answer


def load_llm(args):
    model_name = args.model.split('/')[-1]
    if 'graphwiz' in args.model.lower():
        tokenizer_path = 'models/Llama-2-7b-chat-hf'
    else:
        tokenizer_path = args.model


    LLM = vllm.LLM(model=args.model,
                tensor_parallel_size=args.num_gpu,
                seed=42, 
                gpu_memory_utilization=0.95,
                max_num_seqs=32,
                tokenizer=tokenizer_path)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if model_name in ["DeepSeek-R1-Distill-Qwen-7B"]:
        max_tokens = 32000
    else:
        max_tokens = 4096
    sampling_params = vllm.SamplingParams(temperature=args.temperature, 
                                        max_tokens=max_tokens, 
                                        top_p=0.95,
                                        top_k=30)
    return LLM, tokenizer, sampling_params


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0


def construct_graph(s):
    edge_pattern = r'\((\d+)->(\d+)\)'
    edges = re.findall(edge_pattern, s)
    edges = [(int(u), int(v)) for u, v in edges]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


def convert_int_to_set(string):
    if string and string.isdigit():
        return f'[{string}]'
    return string


def is_a_list_tuple(obj):
    return obj and type(obj) in [list, tuple]


def is_valid_node_set(node_set):
    if is_a_list_tuple(node_set):
        for ele in node_set:
            if type(ele) != int:
                return False
        return True
    return False


def is_valid_topological_sort(graph, node_set):
    if not is_valid_node_set(node_set):
        return False
    
    if set(node_set) != set(graph.nodes()):
        return False
    
    position = {node: idx for idx, node in enumerate(node_set)}
    for u, v in graph.edges():
        if position[u] >= position[v]:
            return False
    return True


def verify_correctness_graphwiz(task, prompt, truth, predict, model_name):    
    golden_answer = truth.replace("###", "").strip()
    generated_answer = extract_solution(predict, model_name)
    
    if not generated_answer or not golden_answer:
        correct = False

    elif task in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        correct = golden_answer.lower() == generated_answer.lower()

    elif task in ['flow', 'shortest', 'triangle']:
        golden_answer = extract_last_num(golden_answer)
        generated_answer = extract_last_num(generated_answer)
        correct = True if abs(golden_answer- generated_answer) < 1e-2 else False

    elif task == 'topology':
        golden_set = list(ast.literal_eval(golden_answer))[0]
        graph = construct_graph(prompt)
        try:
            answer_set = list(ast.literal_eval(convert_int_to_set(generated_answer)))
            correct = is_valid_topological_sort(graph, answer_set)
        except:
            print(f'Node list conversion errors {generated_answer}, {golden_answer}')
            correct = False
    else:
        raise ValueError

    return correct, generated_answer


def answer_format(task):
    if task in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        return "\nYour answer should be Yes or No."
    if task in ['flow', 'shortest', 'triangle']:
        return "\nYou need to format your answer as a float number."
    if task == 'topology':
        return "\nYou need to format your answer as a list of nodes, e.g., [node-1, node-2, ..., node-n]."
    raise ValueError


def evaluate_single_task(task, args, LLM, tokenizer, sampling_params):
    dataset = datasets.load_dataset(f"data/GraphWiz-Revised", split=task)
    prompts = dataset['input_prompt']
    answers = dataset['answer']

    dataset_size = len(prompts)
    print(f"Test number for {task}: {dataset_size}")

    outputs = []
    if "graphwiz" not in args.model.lower():
        test_cases = [
            tokenizer.apply_chat_template(
                [{
                    'role':'user',
                    'content': QUERY_TEMPLATE.format(instruction=prompt + answer_format(task))
                }],
                tokenize=False,
                add_generation_prompt=True if "graphwiz" not in args.model.lower() else False,
            ) for prompt in prompts
        ]
    else:
        test_cases = prompts
    
    LLM.llm_engine.tokenizer.tokenizer.truncation_side="left"
    outputs = LLM.generate(test_cases, sampling_params)

    result = []
    for index, output in enumerate(outputs):
        response = output.outputs[0].text 
        correct, generated_answer = verify_correctness_graphwiz(task=task,
                                                                prompt=prompts[index],
                                                                truth=answers[index],
                                                                predict=response,
                                                                model_name=args.model)
        result.append({
            'query': prompts[index],
            'golden_answer': answers[index],
            'generated_answer': generated_answer,
            'generation': response,
            'correct': correct
        })
    return result
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating acc")
    parser.add_argument('--model', type=str, default='models/Qwen2.5-3B-Instruct')
    parser.add_argument('--num_gpu', type=int, default=2)
    parser.add_argument('--save_path', type=str, default='eval_results/graphwiz')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--seed', type=float, default=42)

    args = parser.parse_args()
    print(args)

    LLM, tokenizer, sampling_params = load_llm(args)
    
    meta_data = {}
    total = 0
    correct = 0
    full_result = {}
    for task in ['topology', 'cycle', 'connectivity', 'flow', 'bipartite', 'hamilton', 'shortest', 'substructure', 'triangle']:
        print(f'Evaluating task [{task}]')
        result = evaluate_single_task(task, args, LLM, tokenizer, sampling_params)
        full_result[task] = result
        
        meta_data[task] = {
            'total': len(result),
            'correct': 0,
            'acc': 0
        }

        for sample in result:
            meta_data[task]['correct'] += sample['correct']
            total += 1
            correct += sample['correct']

        meta_data[task]['acc'] = meta_data[task]['correct'] / meta_data[task]['total']
        
    meta_data['average'] = {
        'total': total,
        'correct': correct,
        'acc': correct / total
    }
    full_result['meta_data'] = meta_data


    ## Save result
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'{args.model.split("/")[-1]}_eval_result.json')
    with open(save_path, 'w') as f:
        json.dump(full_result, f, indent=4, ensure_ascii=False)