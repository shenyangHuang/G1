import sys, os, json, argparse, re, ast, vllm, datasets
from collections import defaultdict
from tasks import *
import numpy as np
from transformers import AutoTokenizer
import networkx as nx

from eval_utils import *


QUERY_TEMPLATE = """
{instruction}

Approach the problem methodically. Ensure all conclusions are based on precise calculations and logical deductions. Feel free to explore various solution methods and cross-check results for consistency. Maintain dynamic thinking and always verify each step of your reasoning.

Present the final answer in \\boxed{{}} format, like this: $\\boxed{{ANSWER}}$, where ANSWER is the final result or expression.

Think carefully and break down the problem step by step.
""".strip()


def convert_int_to_set(string):
    if string and string.isdigit():
        return f'[{string}]'
    return string


def extract_solution(solution_str, model_name):
    if 'graphwiz' in model_name.lower():
        last_index = solution_str.rfind("###")
        generated_answer = solution_str[last_index + 3:] if last_index != -1 else None
    else:
        possible_ans = last_boxed_only_string(solution_str)
        generated_answer = remove_boxed(possible_ans).strip() if possible_ans else None
    
    if generated_answer is None:
        pattern = r"(?:the (?:final )?answer is\s*)(.*)"
        match = re.search(pattern, solution_str, re.IGNORECASE)
        if match:
            generated_answer = match.group(1).strip()
            generated_answer = generated_answer.replace(":", "").replace("\\n", "").replace(".", "")
        else:
            generated_answer = None
    return generated_answer


def load_llm(args):
    LLM = vllm.LLM(model=args.model, 
                tensor_parallel_size=args.num_gpu, 
                seed=42, 
                gpu_memory_utilization=0.9,
                tokenizer=args.tokenizer,
                max_num_seqs=32)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    sampling_params = vllm.SamplingParams(temperature=args.temperature, 
                                        max_tokens=args.max_length, 
                                        top_p=0.95,
                                        top_k=30)    
    return LLM, tokenizer, sampling_params



def generate_response(prompts, LLM, tokenizer, sampling_params):
    outputs = []
    test_cases = [
        tokenizer.apply_chat_template(
            [{
                'role':'user',
                'content': QUERY_TEMPLATE.format(instruction=prompt)
            }],
            tokenize=False,
            add_generation_prompt=True, # False for GraphWiz-DPO and GraphWiz
        ) for prompt in prompts
    ]
    LLM.llm_engine.tokenizer.tokenizer.truncation_side="left"
    outputs = LLM.generate(test_cases, sampling_params)
    return outputs


def clean(ans):
    if ans is None:
        ans = ''
    return ans.replace('text', '').replace('\\', '').replace('{', '').replace('}', '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating GraphArena")
    parser.add_argument('--model', type=str, default='../models/Qwen2.5-3B-Instruct')
    parser.add_argument('--num-gpu', type=int, default=2)
    parser.add_argument('--save-path', type=str, default='eval_result')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--max-length', type=int, default=4096)
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model

    print(args)

    dataset_path = "./dataset"
    with open(f'{dataset_path}/0shot.json', 'r') as f: ## Customize: modify to evaluate on few shot settings
        data_dict = json.load(f)
    problem_num = 100
    LLM, tokenizer, sampling_params = load_llm(args)

    meta_data = {}
    full_result = {}
    less_is_better = ['GED', 'TSP', 'MVC', 'Distance'] 
    
    for task_name in ['Connected', 'Distance', 'Diameter', 'GED', 'MCP', 'MCS', 'MIS', 'MVC', 'Neighbor', 'TSP']:  ## Customize: choose what tasks to evaluate
        for difficulty in ['easy', 'hard']: ## Customize: choose what difficulties to evaluate

            print(f'Evaluating task [{task_name}], difficulty [{difficulty}]')
            task = globals()[task_name + '_Task'](dataset_path)
            task.load_dataset(difficulty)
            prompts = [task.problem_set[i]["problem_text"] for i in range(0, problem_num)]
            outputs = generate_response(prompts, LLM, tokenizer, sampling_params)

            result = []
            metrics = {'rank':[], 'feasible':[], 'MRR':[], 'hallu':[], 'acc': [],'top1':[], 'top3':[], 'len': []}
            for i in range(0, problem_num):
                response = outputs[i].outputs[0].text
                generated_answer = extract_solution(response, args.model)
                generated_answer = clean(generated_answer)
                
                gt = task.problem_set[i]['exact_answer']
                if gt is None:
                    try:
                        gt = task.problem_set[i]['approx_answer']
                    except:
                        gt = None
                
                if generated_answer is None:
                    generated_answer = ""

                if '[' in prompts[i] and '[' not in generated_answer:
                    generated_answer = f'[{generated_answer}]'
                    
                pred = task.check_solution(i, generated_answer)

                metrics['feasible'].append(pred > 0)
                metrics['hallu'].append(pred == -2)
                if gt is not None:
                    if task_name in ['GED', 'TSP', 'MVC']:
                        acc = bool(0 <= pred and pred <= gt)
                    elif task_name in ['MCP', 'MCS', 'MIC']:
                        acc = bool(pred >= gt)
                    else:
                        acc = bool(pred == gt)
                else:
                    acc = False
                metrics['acc'].append(acc)

                result.append({
                    'query': prompts[i],
                    'golden_answer': gt,
                    'generated_answer': generated_answer,
                    'generation': response,
                    'correct': acc
                })
            
            tmp_save_path = os.path.join(args.save_path, f'{args.model.split("/")[-1]}')
            os.makedirs(tmp_save_path, exist_ok=True)
            with open(tmp_save_path+f'/{task_name}_{difficulty}.json', 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            avg_feasible = sum(metrics['feasible']) / problem_num
            avg_hallu = sum(metrics['hallu']) / problem_num
            avg_acc = sum(metrics['acc']) / problem_num

            full_result[task_name + "_" + difficulty] = result
            meta_data[task_name + "_" + difficulty] = {
                'total': problem_num,
                'acc': avg_acc
            }


    avg_task_acc = 0
    for task in list(meta_data.keys()):
        avg_task_acc += meta_data[task]['acc']
    avg_task_acc /= len(list(meta_data.keys()))
    meta_data['average'] = avg_task_acc

    full_result['meta_data'] = meta_data

    ## Save result
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'{args.model.split("/")[-1]}_eval_result.json')
    with open(save_path, 'w') as f:
        json.dump(full_result, f, indent=4, ensure_ascii=False)
