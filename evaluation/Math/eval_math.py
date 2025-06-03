from transformers import AutoTokenizer
import json, os, argparse, re, vllm, datasets, ast
from eval_utils import *


def further_clean(s):
    return s.replace(' ', '').replace(',', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '').replace('$', '').replace('\'', '').replace('\\', '').replace('−', '-').replace('times', '×').replace('~', '').replace('mathrm', '')

def equal_function(generated_answer, golden_answer):
    label = False
    
    if not generated_answer or not golden_answer:
        return generated_answer, golden_answer, label

    label = (further_clean(generated_answer) == further_clean(golden_answer))
    return generated_answer, golden_answer, label


MATH_QUERY_TEMPLATE = """
    {instruction}

    Approach the problem methodically. Ensure all conclusions are based on precise calculations and logical deductions. Feel free to explore various solution methods and cross-check results for consistency. Maintain dynamic thinking and always verify each step of your reasoning.

    Present the final answer in \\boxed{{}} format, like this: $\\boxed{{ANSWER}}$, where ANSWER is the final result or expression.

    Think carefully and break down the problem step by step.
    """.strip()


parser = argparse.ArgumentParser(description="Evaluating acc")
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
    

    sampling_params = vllm.SamplingParams(temperature=args.temperature, 
                                        max_tokens=args.max_length, 
                                        top_p=0.95,
                                        top_k=30)
        

    LLM = vllm.LLM(model=args.model, tensor_parallel_size=args.num_gpu, seed=20021201, gpu_memory_utilization=0.6, tokenizer=tokenizer_name, max_num_seqs=128)

    result = {}
    ### Evaluating Math
    eval_math_dataset = datasets.load_dataset(f"CharlesLi/graph_hard", split='test')
    eval_math_dataset = eval_math_dataset.filter(lambda x : 'graph' not in x['tasks'])
    prompts = eval_math_dataset['prompts']
    answers = eval_math_dataset['answers']
    tasks = eval_math_dataset['tasks']
    test_cases = [
        tokenizer.apply_chat_template(
            [{
                'role':'user',
                'content': MATH_QUERY_TEMPLATE.format(instruction = ins)
            }], tokenize=False
        ) for ins in prompts
    ]
    outputs = LLM.generate(test_cases, sampling_params)
    index = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_answer = extract_solution(generated_text)
        golden_answer = strip_string(str(answers[index]))

        info = {
            "task": tasks[index]
        }

        generated_answer, golden_answer, label = equal_function(generated_answer, golden_answer)

        if tasks[index] not in result:
            result[tasks[index]] = []
        
        result[tasks[index]].append(
            {
                'query': prompts[index],
                'golden_answer': answers[index],
                'generated_answer': generated_answer,
                'generation': generated_text,
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

    
    meta_data['MATH_acc'] = {
            'total': 0,
            'correct': 0,
            'acc': 0
        }
            
    keys = []
    for key, item in meta_data.items():
        if 'level' in key.lower():
            meta_data['MATH_acc']['total'] += item['total']
            meta_data['MATH_acc']['correct'] += item['correct']

    meta_data['MATH_acc']['acc'] = meta_data['MATH_acc']['correct'] / meta_data['MATH_acc']['total']

    result['meta_data'] = meta_data

    ## Save result
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'eval_result.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
