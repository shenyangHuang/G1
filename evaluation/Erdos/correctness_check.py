import re
import ast
from networkx.algorithms import approximation
from collections import defaultdict, deque

from algorithms import *
from string_utils import strip_string
from math_utils import remove_boxed, last_boxed_only_string


def verify_correctness(response, ground_truth, graphs, samples, answer_type, task_name, model_name):        
    generated_answer = extract_solution(response, model_name)
    golden_answer = extract_answer(ground_truth)
    correct = equal_function(graphs, samples, generated_answer, golden_answer, answer_type, task_name)
    return correct, generated_answer


def convert_int_to_set(string):
    if string and string.isdigit():
        return f'[{string}]'
    return string


def extract_answer(ground_truth):
    extracted_answer = strip_string(str(ground_truth))
    return extracted_answer
    

def extract_solution(solution_str, model_name):
    if 'graphwiz' in model_name.lower():
        last_index = solution_str.rfind("###")
        generated_answer = solution_str[last_index + 3:].strip() if last_index != -1 else None
    else:
        possible_ans = last_boxed_only_string(solution_str)
        generated_answer = strip_string(remove_boxed(possible_ans)) if possible_ans else None
    
    if generated_answer is None:
        pattern = r"(?:the (?:final )?answer is\s*)(.*)"
        match = re.search(pattern, solution_str, re.IGNORECASE)
        if match:
            generated_answer = match.group(1).strip()
            generated_answer = generated_answer.replace(":", "").replace("\\n", "").replace(".", "")
        else:
            generated_answer = None
    return generated_answer
    

def extract_letters(input_string):
    return ''.join(re.findall(r'[a-zA-Z]', input_string))


def extract_floats(input_string):
    s = input_string.strip()
    try:
        value = float(s)
        return value
    except ValueError:
        pass

    frac_match = re.match(r'frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}', s)
    if frac_match and len(frac_match.groups()) >= 2:
        numerator = float(frac_match.group(1))
        denominator = float(frac_match.group(2))
        if denominator != 0:
            return numerator / denominator
        else:
            return None  # 分母为零非法

    div_match = re.match(r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', s)
    if div_match and len(div_match.groups()) >= 2:
        numerator = float(div_match.group(1))
        denominator = float(div_match.group(2))
        if denominator != 0:
            return numerator / denominator
        else:
            return None  # 分母为零非法
    return None


def equal_function(graphs, samples, generated_answer, golden_answer, answer_type, task_name):
    if not generated_answer or not golden_answer:
        return False

    if generated_answer == golden_answer:
        return True

    if answer_type == 'integer':
        return generated_answer == golden_answer
    
    elif answer_type == "bool":
        golden_answer = extract_letters(golden_answer.lower())
        generated_answer = extract_letters(generated_answer.lower())
        return golden_answer == generated_answer
    
    
    elif answer_type == "float":
        golden_number = extract_floats(golden_answer)
        generated_number = extract_floats(generated_answer)

        if golden_number is not None and generated_number is not None:
            return (golden_number - generated_number)**2 < 1e-6
        else:
            print(f'Float number evaluation errors {generated_answer}, {golden_answer}')
            return False
        
        
    elif answer_type in ["ordered_node_list", "node_list"]:        
        golden_set = list(ast.literal_eval(convert_int_to_set(golden_answer)))
        try:
            answer_set = list(ast.literal_eval(convert_int_to_set(generated_answer)))
        except:
            print(f'Node list conversion errors {generated_answer}, {golden_answer}')
            return False

        if task_name == 'dominating_set':
            return is_valid_dominating_set(graphs[0], answer_set)
        
        if task_name == 'maximal_independent_set':
            return is_maximal_independent_set(graphs[0], answer_set)
        
        if task_name == 'min_vertex_cover':
            return is_minimum_vertex_covering(graphs[0], answer_set)
        
        if task_name in ['shortest_path', 'weighted_shortest_path']:
            samples = ast.literal_eval(samples)
            return is_valid_shortest_path(graphs[0], answer_set, samples[0], samples[1])

        if task_name == 'topological_sort':
            return is_valid_topological_sort(graphs[0], answer_set)
        
        return golden_set  == answer_set

        
    elif answer_type in ["ordered_edge_list", "edge_list"]:
        golden_set = list(ast.literal_eval(convert_int_to_set(golden_answer)))
        try:
            answer_set = list(ast.literal_eval(convert_int_to_set(generated_answer)))
        except:
            print(f'Edge list conversion errors {generated_answer}, {golden_answer}')
            return False
        
        if task_name == 'min_edge_covering':
            return is_minimum_edge_covering(graphs[0], answer_set)
        
        if task_name == 'minimum_spanning_tree':
            return is_minimum_spanning_tree(graphs[0], answer_set)
        
        if task_name == 'weighted_minimum_spanning_tree':
            return is_minimum_spanning_tree_weighted(graphs[0], answer_set)
        
        if task_name == 'max_weight_matching':
            return is_maximal_weight_matching(graphs[0], answer_set)
        
        if task_name == 'bipartite_maximum_matching':
            return is_bipartite_maximum_matching(graphs[0], answer_set)
        
        if task_name == 'bfs':
            return is_valid_bfs_sequence(graphs[0], answer_set, start_node=int(samples))
        
        if task_name == 'dfs':
            return is_valid_dfs_sequence(graphs[0], answer_set, start_node=int(samples))
        
        return golden_set == answer_set
    
    
    elif answer_type == "dict":
        golden_dict = ast.literal_eval(golden_answer)
        try:
            answer_dict = ast.literal_eval(generated_answer)
        except:
            print(f'Dict conversion errors {generated_answer}, {golden_answer}')
            return False
        if task_name == 'isomophic_mapping':
            return is_valid_isomorphism(graphs[0], graphs[1], answer_dict)
        
        return golden_dict == answer_dict



if __name__ == "__main__":
    d = str({1: 101, 2: 102})
    d = ast.literal_eval(d)
    print(d)
    print(type(d))
    
