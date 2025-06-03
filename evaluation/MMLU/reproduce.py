import json, os

def furthur_clean(s):
    return s.replace(' ', '').replace(',', '').replace('.', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '').replace('$', '').replace('\'', '').replace('\\', '').replace('−', '-').replace('times', '×').replace('~', '').replace('mathrm', '').replace('div', '/').replace('frac', '').replace('[', '').replace(']', '').replace('%', '').replace('—', '-').replace('sqrt', '').replace('π', 'pi').replace('.0', '')

def equal_function(generated_answer, golden_answer):
    label = False
    
    if not generated_answer or not golden_answer:
        return generated_answer, golden_answer, label

    label = (furthur_clean(generated_answer) == furthur_clean(golden_answer))
    return label


for path in [
    "original_logs/Llama-3.2-3B-Instruct_eval_result.json",
    "original_logs/Qwen2.5-3B-Instruct_eval_result.json",
    "original_logs/CoT-SFT-3B_eval_result.json",
    "original_logs/G1-3B_eval_result.json",
    "original_logs/Llama-3.1-8B-Instruct_eval_result.json",
    "original_logs/DeepSeek-R1-Distill-Qwen-7B_eval_result.json",
    "original_logs/Qwen2.5-7B-Instruct_eval_result.json",
    "original_logs/CoT-SFT-7B_eval_result.json",
    "original_logs/G1-7B_eval_result.json",
]:
    old_result = json.load(open(path))
    new_result = {}

    def convert_to_int(s):
        if s.isdigit():  
            return int(s)
        elif len(s) == 1 and 'A' <= s <= 'Z': 
            return ord(s) - ord('A') 
        else:
            return 1e6

    for key, items in old_result.items():
        if key == 'meta_data':
            continue

        new_result[key] = items
        for i, item in enumerate(items):

            ## There are cases in which the models fail to follow the answer format precisely.
            ## We specially handle some cases where the correctness of the responses are unnecessarily influenced by strings like "mathrm" and "times".
            ## Excluding this will not invalidate our claims.
            
            if not item["label"] and item["generated_answer"]:
                convert_int = convert_to_int(item["generated_answer"])

                if convert_int >= 0 and convert_int < len(item["info"]["sample"]):
                    generated_answer = item["info"]["sample"][convert_int]
                    new_result[key][i]["generated_answer"] = generated_answer

                    new_result[key][i]["label"] = equal_function(
                        generated_answer,
                        item["golden_answer"]
                    )

    meta_data = {}

    for key in new_result:
        meta_data[key] = {
            'total': len(new_result[key]),
            'correct': 0,
            'acc': 0
        }

        for sample in new_result[key]:
            meta_data[key]['correct'] += sample['label']

        meta_data[key]['acc'] = meta_data[key]['correct'] / meta_data[key]['total']

    meta_data['MMLU_acc'] = {
            'total': 0,
            'correct': 0,
            'acc': 0
        }
    print(f"{path.split('/')[-1]}", end = ' ')

    for key, item in meta_data.items():
        if 'mmlu' not in key.lower():
            meta_data['MMLU_acc']['total'] += item['total']
            meta_data['MMLU_acc']['correct'] += item['correct']

            print(f"& {100*item['correct']/item['total']:.2f}", end=' ')

    meta_data['MMLU_acc']['acc'] = meta_data['MMLU_acc']['correct'] / meta_data['MMLU_acc']['total']

    print(f"& {100*meta_data['MMLU_acc']['acc']:.2f}", end='\\\\')
    print('\n\\midrule')

    new_result['meta_data'] = meta_data