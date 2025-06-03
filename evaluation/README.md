# Evaluation
 

We provide evaluation scripts for multiple benchmarks:
 | Benchmark Type	| Included Tests |
 |--------------|------------------|
 | Graph Reasoning | Erdős, [GraphWiz](https://dl.acm.org/doi/10.1145/3637528.3672010), [GraphArena](https://openreview.net/forum?id=Y1r9yCMzeA) |
 | Graph Real-World Tasks |  [Node Classification](https://arxiv.org/abs/2502.18771), [Link Prediction](https://arxiv.org/abs/2502.18771) |
 | General Reasoning | [GSM8K](https://arxiv.org/abs/2110.14168), [MATH](https://arxiv.org/abs/2103.03874), [MMLU-pro](https://arxiv.org/abs/2406.01574) |



## 🍥 Erdős

### Dataset Preparation (optional)

Download the Erdős dataset from [PKU-ML/Erdos](https://huggingface.co/datasets/PKU-ML/Erdos).

### Running Evaluation

Execute the following command to evaluate models on Erdős:

```bash
cd Erdos
python eval_graphwiz.py \
    --data_path PKU-ML/Erdos \
    --model PKU-ML/G1-7B  \
    --num_gpu 1 \
    --save_path eval_results/erdos \
    --temperature 0.6
```
- `data_path`: Local path (like `data/erdos`) or HuggingFace dataset ID (`PKU-ML/Erdos`)
- `model`: Path to model in HuggingFace format.
- `num_gpu`: Number of GPUs for parallel inference.
- `save_path`: Directory to save evaluation logs (includes: queries, model outputs, ground truths, and per-task accuracy).
- `temperature`: Sampling temperature for vLLM (fixed at 0.6 for all experiments).

### Reproducibility
We open-source the complete experiment logs in [GoogleDrive](https://drive.google.com/drive/folders/1fWtPkhIey98IqW7GeND28X_7rjoJEuxh?usp=sharing). One can download the logs locally and reproduce the numbers reported in our paper by running 

```bash
cd Erdos
python reproduce.py > reproduce.txt
```



## 🍥 GraphWiz

### Download benchmark (optional)
Download the revised GraphWiz dataset from [PKU-ML/GraphWiz-Revised](https://huggingface.co/datasets/PKU-ML/GraphWiz-Revised).

> **Important Note**:
> We identified approximately 60% incorrect ground truths in the original GraphWiz's Hamilton Path task.
> Our revised version contains corrected annotations and has been fully re-uploaded for accurate evaluation.

### Running Evaluation
Execute the following command to evaluate models on GraphWiz:

```bash
cd GraphWiz
python eval_graphwiz.py \
    --data_path PKU-ML/GraphWiz-Revised \
    --model PKU-ML/G1-7B  \
    --num_gpu 1 \
    --save_path eval_results/graphwiz  \
    --temperature 0.6
```
- `data_path`: Local path (like `data/graphwiz`) or HuggingFace dataset ID (`PKU-ML/GraphWiz-Revised`)
- `model`: Path to model in HuggingFace format.
- `num_gpu`: Number of GPUs for parallel inference.
- `save_path`: Directory to save evaluation logs (includes: queries, model outputs, ground truths, and per-task accuracy).
- `temperature`: Sampling temperature for vLLM (fixed at 0.6 for all experiments).

### Reproducibility
We open-source the complete experiment logs in [GoogleDrive](https://drive.google.com/drive/folders/1fWtPkhIey98IqW7GeND28X_7rjoJEuxh?usp=sharing). One can download the logs locally and reproduce the numbers reported in our paper by running 

```bash
cd GraphWiz
python reproduce.py > reproduce.txt
```

### Acknowledgement
- [GraphWiz Paper](https://dl.acm.org/doi/10.1145/3637528.3672010)
- [GraphWiz Repo](https://github.com/nuochenpku/Graph-Reasoning-LLM)



## 🍥 GraphArena
To evaluate LLMs on the GraphArena benchmark, one needs to first configure the dataset following the original repo, which we specify below. 

### Environment Setup
```bash
conda create -n GraphArena
source activate GraphArena
conda install openai pandas numpy networkx pip
pip install pybind11
pip install rdkit ogb graph-walker fast_tsp munkres
```

### Dataset Preparation
Download and unzip `dataset.zip` from the [google drive](https://drive.google.com/drive/folders/1mvJSUTrfOX13wgpkyb3w8s_SJqipnb1c?usp=sharing), which contains the processed dataset.  
To build the dataset from scratch, download `source.zip` from the same link and run `bash utils/build_dataset.sh`.

### Running Evaluation
Below we provide an example of starting the evaluation, which will save results to ```eval_results/G1-7B```. One can customize what tasks/diffculties to evaluate on and whether to use few-shot examples by modifying line 107/116/177 of ```eval_grapharena.py```.

```bash
cd GraphArena
python eval_grapharena.py \
    --model PKU-ML/G1-7B \
    --save-path 'eval_results/grapharena' \
    --num-gpu 2 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.6 \
    --max-length 4096 # Maximum response length
```

### Reproducibility
We open-source the complete experiment logs in [GoogleDrive](https://drive.google.com/drive/folders/1fWtPkhIey98IqW7GeND28X_7rjoJEuxh?usp=sharing). One can download the logs locally and reproduce the numbers reported in our paper by running 

```bash
cd GraphArena
python reproduce.py > reproduce.txt
```

### Acknowledgement
- [GraphArena Paper](https://openreview.net/forum?id=Y1r9yCMzeA)
- [GraphArena Repo](https://github.com/squareRoot3/GraphArena)



## 🍥 Node Classification and Link Prediction

### Dataset Preparation (optional)
Download the datasets from [PKU-ML/node_classification](https://huggingface.co/datasets/PKU-ML/node_classification) and [PKU-ML/link_prediction](https://huggingface.co/datasets/PKU-ML/link_prediction).

### Running Evaluation
Execute the following command to evaluate models on Node Classification and Link Prediction:

```bash
cd RealWorld
python eval_node_classification.py \
    --data_path PKU-ML/node_classification \
    --model PKU-ML/G1-7B  \
    --num_gpu 1 \
    --save_path eval_results/node_classification  \
    --temperature 0.6
```
   
```bash
cd RealWorld
python eval_link_prediction.py \
    --data_path PKU-ML/link_prediction \
    --model PKU-ML/G1-7B  \
    --num_gpu 1 \
    --save_path eval_results/link_prediction  \
    --temperature 0.6
```

- `data_path`: Local path (like `data/node_classification`) or HuggingFace dataset ID (like `PKU-ML/node_classification`)
- `model`: Path to model in HuggingFace format.
- `num_gpu`: Number of GPUs for parallel inference.
- `save_path`: Directory to save evaluation logs (includes: queries, model outputs, ground truths, and per-task accuracy).
- `temperature`: Sampling temperature for vLLM (fixed at 0.6 for all experiments).

### Reproducibility
We open-source the complete experiment logs in [GoogleDrive](https://drive.google.com/drive/folders/1fWtPkhIey98IqW7GeND28X_7rjoJEuxh?usp=sharing). One can download the logs locally and reproduce the numbers reported in our paper by running 

```bash
cd RealWorld
python reproduce.py > reproduce.txt
```

### Acknowledgement
- [Benchmark Paper](https://arxiv.org/abs/2502.18771)
- [Benchmark Repo](https://github.com/myflashbarry/LLM-benchmarking)




## 🍥 GSM8K and MATH-500

### Running Evaluation
Below we provide an example of starting the evaluation, which will save results to ```eval_results/G1-7B```. 

```bash
cd Math
python eval_math.py \
    --model PKU-ML/G1-7B \
    --save-path 'eval_results' \
    --num-gpu 2 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.6 \
    --max-length 4096 # Maximum response length
```

### Reproducibility
We open-source the complete experiment logs in [GoogleDrive](https://drive.google.com/drive/folders/1fWtPkhIey98IqW7GeND28X_7rjoJEuxh?usp=sharing). One can download the logs locally and reproduce the numbers reported in our paper by running 

```bash
cd Math
python reproduce.py > reproduce.txt
```

### Acknowledgement
- [GSM Paper](https://arxiv.org/abs/2110.14168)
- [GSM Repo](https://github.com/openai/grade-school-math)
- [MATH-500 Repo](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)




## 🍥 MMLU-pro

### Running Evaluation
Below we provide an example of starting the evaluation, which will save results to ```eval_results/G1-7B```. 

```bash
cd MMLU
python eval_mmlu.py \
    --model PKU-ML/G1-7B \
    --save-path 'eval_results' \
    --num-gpu 2 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.6 \
    --max-length 4096 # Maximum response length
```

### Reproducibility
We open-source the complete experiment logs in [GoogleDrive](https://drive.google.com/drive/folders/1fWtPkhIey98IqW7GeND28X_7rjoJEuxh?usp=sharing). One can download the logs locally and reproduce the numbers reported in our paper by running 

```bash
cd MMLU
python reproduce.py > reproduce.txt
```

### Acknowledgement
- [MMLU-Pro Paper](https://arxiv.org/abs/2406.01574)
- [MMLU Repo](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro)

  
