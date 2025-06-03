from transformers import AutoTokenizer
import json
import os
import argparse
import vllm
import datasets
from collections import defaultdict

from math_utils import remove_boxed, last_boxed_only_string


def load_model(args):
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


def generate_response(LLM, tokenizer, prompts):
    if "graphwiz" not in args.model.lower():
        test_cases = [
            tokenizer.apply_chat_template(
                [{
                    'role':'user',
                    'content': QUERY_TEMPLATE.format(instruction = ins)
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


QUERY_TEMPLATE = """
{instruction}

The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$.' (without quotes) where ANSWER is just the final answer. Think step by step before answering.
""".strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluating acc")
    parser.add_argument('--data_path', type=str, default="data/GraphWiz-Revised")
    parser.add_argument('--model', type=str, default='models/Qwen2.5-3B-Instruct')
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='eval_results/node_classification')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--seed', type=float, default=42)
    args = parser.parse_args()
    print(args)

    data_path = args.data_path

    LLM, tokenizer, sampling_params = load_model(args)

    result = defaultdict(list)
    for data_name in ['cora', 'pubmed']:
        for task in ['with_label', 'without_label']:
            task_name = data_name + '_' + task

            dataset = datasets.load_dataset(data_path, split=task_name)
            prompts = dataset['prompt']
            answers = dataset['answer']
            
            print(f"Test number: {len(prompts)}")
            outputs = generate_response(LLM, tokenizer, prompts)

            for index, output in enumerate(outputs):
                answer = answers[index]
                generation = output.outputs[0].text
                possible_ans = last_boxed_only_string(generation)
                if possible_ans is not None:
                    generated_answer = remove_boxed(possible_ans)
                    generated_answer = generated_answer.replace("\\", "").replace("{", "").replace("}", "").replace("text", "")
                else:
                    generated_answer = None
                
                if generated_answer is not None:
                    correct = generated_answer.strip().lower() == answer.strip().lower()
                else:
                    correct = False
                
                result[task_name].append(
                    {
                        'query': prompts[index],
                        'golden_answer': answers[index],
                        'generated_answer': generated_answer,
                        'generation': generation,
                        'correct': correct
                    }
                )
        
        
    meta_data = {}
    total = 0
    correct = 0
    for task in result:
        meta_data[task] = {
            'total': len(result[task]),
            'correct': 0,
            'acc': 0
        }

        for sample in result[task]:
            meta_data[task]['correct'] += sample['correct']
            total += 1
            correct += sample['correct']

        meta_data[task]['acc'] = meta_data[task]['correct'] / meta_data[task]['total']
        
    meta_data['average'] = {}
    meta_data['average']['total'] = total
    meta_data['average']['correct'] = correct
    meta_data['average']['acc'] = correct / total

    result['meta_data'] = meta_data

    ## Save result
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'{args.model.split("/")[-1]}_eval_result.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)