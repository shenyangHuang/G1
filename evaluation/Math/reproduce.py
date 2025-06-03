import json, os

for model in [
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-3B-Instruct",
    "CoT-SFT-3B", 
    "G1-3B",

    "Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B", 
    "Qwen2.5-7B-Instruct",
    "CoT-SFT-7B", 
    "G1-7B"
]:
    result = json.load(open(f"original_logs/{model}_eval_result.json"))
    print(model, end = ' ')
    print(f"GSM = {result['meta_data']['GSM8K']}, MATH = {result['meta_data']['MATH_acc']}")