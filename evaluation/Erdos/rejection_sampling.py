from transformers import AutoTokenizer
import datasets
import json
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" 
import argparse
import vllm
from collections import defaultdict
import random

from evaluation.erdos.graph_utils import construct_graph
from evaluation.erdos.correctness_check import verify_correctness


QUERY_TEMPLATE = """
{instruction}

Solve the above problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$' where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.
""".strip()


def create_balanced_subset(data_path, samples_per_task):
    dataset = datasets.load_dataset(data_path, split='train')

    task_groups = defaultdict(list)
    for idx, example in enumerate(dataset):
        task_groups[example['task']].append(idx)
    
    sampled_indices = []
    for task, indices in task_groups.items():
        n_samples = min(samples_per_task, len(indices))
        sampled_indices.extend(random.sample(indices, n_samples))
    
    subset = dataset.select(sampled_indices)
    return subset

    
def generate_outputs(dataset, n_sample):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    LLM = vllm.LLM(model=args.model, 
                   tensor_parallel_size=args.num_gpu, 
                   seed=42, 
                   gpu_memory_utilization=0.95, 
                   tokenizer=args.model,
                   max_num_seqs=16,
                   enforce_eager=False)
    LLM.llm_engine.tokenizer.tokenizer.truncation_side="left"

    outputs = []
    test_cases = []
    size = len(dataset)
    for index in range(0, size):
        test_cases.extend([
            tokenizer.apply_chat_template(
                [{
                    'role':'user',
                    'content': QUERY_TEMPLATE.format(instruction=dataset['prompt'][index]),
                }], 
                tokenize=False,
                add_generation_prompt=True,
            ) for _ in range(n_sample)
        ])
    sampling_params = vllm.SamplingParams(temperature=args.temperature,
                                          max_tokens=2048, 
                                          top_p=0.95,
                                          top_k=30)
    outputs = LLM.generate(test_cases, sampling_params)
    print(f'End generating {len(outputs)} response')
    return outputs


def analyze_outputs(outputs, n_sample):
    task_data = defaultdict(list)

    questions = []
    statistics = {
        'total': defaultdict(list),
        'correct': defaultdict(list)
    }
    find_correct = False
    for index, output in enumerate(outputs):
        question_index = index // n_sample
        if question_index not in questions:
            questions.append(question_index)
            statistics['total'][dataset['task'][question_index]].append(question_index)
            find_correct = False

        response = output.outputs[0].text
        graphs = construct_graph(nodes=dataset['nodes'][question_index], 
                                 edges=dataset['edges'][question_index], 
                                 direction=dataset['direction'][question_index])
        
        correct, generated_answer = verify_correctness(response=response, 
                                                       ground_truth=dataset['answer'][question_index], 
                                                       graphs=graphs, 
                                                       samples=dataset['samples'][question_index], 
                                                       answer_type=dataset['answer_type'][question_index], 
                                                       task_name=dataset['task'][question_index],
                                                       model_name=args.model)
        if correct and not find_correct:
            find_correct = True
            statistics['correct'][dataset['task'][question_index]].append(question_index)
            task_data['answer'].append(response)
            task_data['ground_truth'].append(dataset['answer'][question_index])
            for key in list(dataset.features.keys()):
                if key not in ['answer', 'ground_truth']:
                    task_data[key].append(dataset[key][question_index])
    
    print("Correct example number:")
    for task, value in statistics['total'].items():
        if task not in statistics['correct'].keys():
            statistics['correct'][task] = []
        print(f"[{task}]: {len(statistics['correct'][task])} / {len(value)}")
    return task_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating acc")
    parser.add_argument('--data_path', type=str, default="data/erdos")
    parser.add_argument('--model', type=str, default='models/Qwen2.5-32B-Instruct')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='data/graph_task_cot')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=float, default=42)
    parser.add_argument('--n_sample', type=int, default=8)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    data_path = args.data_path
    samples_per_task = 1000
    dataset = create_balanced_subset(data_path, samples_per_task)

    outputs = generate_outputs(dataset, n_sample=args.n_sample, start_index=args.start_index, end_index=args.end_index)
    task_data = analyze_outputs(outputs, n_sample=args.n_sample, start_index=args.start_index)
    
    output_path = args.save_path + '.json'
    with open(output_path, 'w') as f:
        json.dump(task_data, f)
    print(f"Save datasets into {output_path}")