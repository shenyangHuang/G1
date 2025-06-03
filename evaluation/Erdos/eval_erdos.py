from transformers import AutoTokenizer
import json
import os
import argparse
import vllm
import argparse
import datasets
from collections import defaultdict

from graph_utils import construct_graph
from correctness_check import verify_correctness


QUERY_TEMPLATE = """
{instruction}

Solve the above problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.' (without quotes), where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.
""".strip()


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



def generate_response(prompts, LLM, tokenizer, sampling_params):

    if "graphwiz" not in args.model.lower():
        test_cases = [
            tokenizer.apply_chat_template(
                [{
                    'role':'user',
                    'content': QUERY_TEMPLATE.format(instruction = ins),
                }], 
                tokenize=False,
                add_generation_prompt=True,
            ) for ins in prompts
        ]
    else:
        test_cases = prompts

    LLM.llm_engine.tokenizer.tokenizer.truncation_side="left"
    outputs = LLM.generate(test_cases, sampling_params)
    return outputs


def filter_length(example):
    tokens = tokenizer.encode(example['prompt'])
    return len(tokens) < 4000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating acc")
    parser.add_argument('--data_path', type=str, default="data/erdos")
    parser.add_argument('--model', type=str, default='models/Qwen2.5-3B-Instruct')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='eval_results/erdos')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--seed', type=float, default=42)
    args = parser.parse_args()
    print(args)

    LLM, tokenizer, sampling_params = load_llm(args)

    data_path = args.data_path

    dataset = datasets.load_dataset(data_path, split='test')
    print("Dataset size:", len(dataset))
    if 'graphwiz' in args.model.lower() or "Qwen2.5-Math-7B-Instruct" in args.model: # max_token limitation
        print("Filtering dataset for max token length...")
        dataset = dataset.filter(filter_length)
        print("Filtered size:", len(dataset))

    prompts = dataset['prompt']
    answers = dataset['answer']
    tasks = dataset['task']
    nodes = dataset['nodes']
    edges = dataset['edges']
    directions = dataset['direction']
    answer_types = dataset['answer_type']
    samples = dataset['samples']

    dataset_size = len(prompts)
    print(f"Test number: {dataset_size}")

    outputs = generate_response(prompts, LLM, tokenizer, sampling_params)

    result = defaultdict(list)
    for index, output in enumerate(outputs):
        response = output.outputs[0].text 
        graphs = construct_graph(nodes[index], edges[index], directions[index], tasks[index]) 
        
        correct, generated_answer = verify_correctness(response=response, 
                                                    ground_truth=answers[index], 
                                                    graphs=graphs, 
                                                    samples=samples[index], 
                                                    answer_type=answer_types[index], 
                                                    task_name=tasks[index],
                                                    model_name=args.model)
        
        result[tasks[index]].append(
            {
                'query': prompts[index],
                'golden_answer': answers[index],
                'generated_answer': generated_answer,
                'generation': response,
                'correct': correct
            }
        )
        
    meta_data = {}
    all_total = 0
    all_correct = 0
    for task in result:
        correct = 0
        for sample in result[task]:
            correct += sample['correct']
            all_correct += sample['correct']
            all_total += 1

        meta_data[task] = {
            'total': len(result[task]),
            'correct': correct,
            'acc': correct / len(result[task])
        }

    
    
    meta_data['average'] = {
        'total': all_total,
        'correct': all_correct,
        'acc': all_correct / all_total
    }

    result['meta_data'] = meta_data

    ## Save result
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'{args.model.split("/")[-1]}_eval_result.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)