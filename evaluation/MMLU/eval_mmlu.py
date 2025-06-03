from transformers import AutoTokenizer
import json, os, argparse, re, vllm, datasets, ast
from eval_utils import *


def further_clean(s):
    return s.replace(' ', '').replace(',', '').replace('.', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '').replace('$', '').replace('\'', '').replace('\\', '').replace('−', '-').replace('times', '×').replace('~', '').replace('mathrm', '')

def equal_function(generated_answer, golden_answer):
    label = False
    
    if not generated_answer or not golden_answer:
        return generated_answer, golden_answer, label

    label = (further_clean(generated_answer) == further_clean(golden_answer))
    return generated_answer, golden_answer, label


build_mmlu_query = """
Problem: {problem}
Choices: {choices}
Select exactly one item from the choices as your final answer and always keep its original form without any modification.
"""

MMLU_QUERY_TEMPLATE = """
    {instruction}

    Approach the problem methodically. Ensure all conclusions are based on precise calculations and logical deductions. Feel free to explore various solution methods and cross-check results for consistency. Maintain dynamic thinking and always verify each step of your reasoning.

    Present the final answer in \\boxed{{}} format, like this: $\\boxed{{ANSWER}}$, where ANSWER is the final result or expression.

    Think carefully and break down the problem step by step.
    """.strip()


parser = argparse.ArgumentParser(description="Evaluating MMLU Acc")
parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--num-gpu', type=int, default=1)
parser.add_argument('--save-path', type=str)
parser.add_argument('--max-length', type=int, default=4096)
parser.add_argument('--temperature', type=float, default=0.6)
args = parser.parse_args()
args.save_path = os.path.join(args.save_path, args.model.split('/')[-1])
print(args)



if __name__ == '__main__':

    tokenizer_name = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset_name = "TIGER-Lab/MMLU-Pro"
    dataset = datasets.load_dataset(f"{dataset_name}", split='test').shuffle()
    prompts = dataset['question']
    answers = dataset['answer_index']
    tasks = dataset['category']
    samples = dataset['options']

    test_cases = [
        tokenizer.apply_chat_template(
            [{
                'role':'user',
                'content': MMLU_QUERY_TEMPLATE.format(
                    instruction = build_mmlu_query.format(
                        problem=ins,
                        choices=option
                    )
                )
            }], tokenize=False
        ) for ins, option in zip(prompts, samples)
    ][0:1000]

    sampling_params = vllm.SamplingParams(temperature=args.temperature, 
                                        max_tokens=args.max_length, 
                                        top_p=0.95,
                                        top_k=30)
        

    LLM = vllm.LLM(model=args.model, tensor_parallel_size=args.num_gpu, seed=42, gpu_memory_utilization=0.9, tokenizer=tokenizer_name, max_num_seqs=128)


    LLM.llm_engine.tokenizer.tokenizer.truncation_side="left"
    outputs = LLM.generate(test_cases, sampling_params)

    result = {}
    index = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_answer = extract_solution(generated_text)


        choices = samples[index]
        golden_answer = strip_string(str(choices[answers[index]]))

        info = {
            "task": tasks[index],
            'sample': samples[index]
        }

        generated_answer, golden_answer, label = equal_function(generated_answer, golden_answer)

        if tasks[index] not in result:
            result[tasks[index]] = []
        
        result[tasks[index]].append(
            {
                'query': prompts[index],
                'generation': generated_text,
                'golden_answer': golden_answer,
                'generated_answer': generated_answer,
                'label': label,
                'info': info
            }
        )
        index += 1


    ## Build Meta Data
    meta_data = {}

    for key in result:
        meta_data[key] = {
            'total': len(result[key]),
            'correct': 0,
            'acc': 0
        }

        for sample in result[key]:
            meta_data[key]['correct'] += sample['label']

        meta_data[key]['acc'] = meta_data[key]['correct'] / meta_data[key]['total']

    meta_data['MMLU_acc'] = {
            'total': 0,
            'correct': 0,
            'acc': 0
        }
            
    for key, item in meta_data.items():
        if 'mmlu' not in key.lower():
            meta_data['MMLU_acc']['total'] += item['total']
            meta_data['MMLU_acc']['correct'] += item['correct']


    meta_data['MMLU_acc']['acc'] = meta_data['MMLU_acc']['correct'] / meta_data['MMLU_acc']['total']

    result['meta_data'] = meta_data

    ## Save result
    save_path = args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"{dataset_name.split('/')[-1]}_eval_result.json")
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
