from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION_TEMPLATE = """
    {instruction}

    Solve the above problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.
    """.strip()

model_name = "PKU-ML-G1-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "The task is to determine the degree centrality of a node in the graph.\n\n"\
        "Degree centrality for a node is the fraction of nodes it is connected to.\n\n"\
        "Here is an undirected graph containing nodes from 1 to 15. The edges are: (1, 15), (15, 11), (2, 3), (2, 6), (3, 6), (3, 7), (6, 7), (6, 8), (7, 8), (7, 14), (4, 10), (10, 5), (10, 12), (8, 14), (8, 9), (12, 11), (12, 13).\n\n"\
        "Question: What is the degree centrality of node 2 in the graph?\n\n"\
        "You need to format your answer as a float number."
messages = [
    {"role": "user", "content": INSTRUCTION_TEMPLATE.format(instruction=prompt)}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096,
    top_p=0.95,
    top_k=30,
    temperature=0.6
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
