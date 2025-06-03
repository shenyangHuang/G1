import re
import os
import datasets

from utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/graph_task')
    parser.add_argument('--data_source', default='data/erdos')
    args = parser.parse_args()

    data_source = args.data_source
    train_dataset = datasets.load_dataset(data_source, split='train')
    test_dataset = datasets.load_dataset(data_source, split='test')

    instruction_following = """
    {instruction}

    Approach the problem methodically. Ensure all conclusions are based on precise calculations and logical deductions. Feel free to explore various solution methods and cross-check results for consistency. Maintain dynamic thinking and always verify each step of your reasoning.

    Present the final answer in \\boxed{{}} format, like this: $\\boxed{{ANSWER}}$, where ANSWER is the final result or expression.

    Think carefully and break down the problem step by step.
    """.strip()

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('prompt')
            question = instruction_following.format(instruction = question_raw)

            solution = example.pop('answer')
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
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
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

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))