import os
import datasets
import argparse

from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', default='data/erdos')
    parser.add_argument('--save_dir', default='data/erdos_sft')
    parser.add_argument('--tokenizer_path', default="models/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    def filter_length(prompt):
        tokens = tokenizer.encode(prompt['prompt'])
        return len(tokens) < 1024

    data_source = args.data_source

    train_dataset = datasets.load_dataset(data_source, split='train').filter(filter_length)
    test_dataset = datasets.load_dataset(data_source, split='test').filter(filter_length)

    instruction_following = """
        {instruction}

        Solve the above math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.
        """.strip()

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('prompt')
            question = instruction_following.format(instruction = question_raw)
            answer = example.pop('answer')

            solution = answer

            task = example.pop('task')
            nodes = example.pop('nodes')
            edges = example.pop('edges')
            samples = example.pop('samples')
            direction = example.pop('direction')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "solution": [{
                    "role": "assistant",
                    "content": "\\boxed{" + solution + "}",
                }],
                "answer": [{
                    "role": "assistant",
                    "content": answer,
                }],
                "ability": "graph",
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'task':task,
                    'nodes': nodes,
                    'edges': edges,
                    'sample': samples,
                    'direction': direction
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    save_dir = args.save_dir

    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))