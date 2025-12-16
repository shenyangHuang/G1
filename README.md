<p align="center">
  <img src="https://github.com/PKU-ML/G1/blob/main/G1_logo.png" alt="G1 Logo" width="250">
</p>

# G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning

<a target="_blank" href="https://arxiv.org/abs/2505.18499">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="#">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>

<a target="_blank" href="https://huggingface.co/collections/PKU-ML/g1-683d659e992794fc99618cf2">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>

<a target="_blank" href="https://huggingface.co/datasets/PKU-ML/Erdos">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-blue?style=flat"></a>

<br>
<span>
<b>Authors:</b> 
<!--   Alex Su<sup>*</sup>, -->
Xiaojun Guo<sup>*</sup>, 
Ang Li<sup>*</sup>, 
<a class="name" target="_blank" href="https://yifeiwang77.com/">Yifei Wang<sup>*</sup></a>,
<a class="name" target="_blank" href="https://people.csail.mit.edu/stefje/">Stefanie Jegelka</a>
  <a class="name" target="_blank" href="https://yisenwang.github.io/">Yisen Wang<sup>&Dagger;</sup></a>
<br>
<sup>*</sup>Equal Contribution. 
<sup>&Dagger;</sup>Correspondence.
</span>


## 📊 Overview

Our dataset contains **50 graph-theoretic tasks** (100k train, 5k test) categorized by difficulty. Below shows the performance comparison between the base model (`Qwen2.5-7B-Instruct`) and our RL-trained **G1-7B** model. **G1-7B** consistently outperforms the base model, with **+37.07% average accuracy gain**.

| Difficulty    | Tasks | Ratio   | Base Model Acc | G1-7B Acc |
|--------------|------------------|---------|----------------|-----------|
| **Easy**     | Node Number, Dominating Set, Common Neighbor, Edge Number, Neighbor, BFS, Has Cycle, DFS, Minimum Spanning Tree, Edge Existence, Is Regular, Degree, Is Tournament, Density | 29.16% | 57.16% | **95.07%** |
| **Medium**   | Adamic Adar Index, Clustering Coefficient, Connected Component Number, Bipartite Maximum Matching, Local Connectivity, Jaccard Coefficient, Min Edge Covering, Is Eularian, Degree Centrality, Is Bipartite, Resource Allocation Index | 22.91% | 42.55% | **88.91%** |
| **Hard**     | Max Weight Matching, Closeness Centrality, Traveling Salesman Problem, Strongly Connected Number, Shortest Path, Center, Diameter, Barycenter, Radius, Topological Sort, Periphery, Betweenness Centrality, Triangles, Average Neighbor Degree, Harmonic Centrality, Bridges | 33.33% | 18.87% | **50.44%** |
| **Challenging** | Isomorphic Mapping, Global Efficiency, Maximal Independent Set, Maximum Flow, Wiener Index, Hamiltonian Path, Min Vertex Cover | 14.58% | 3.29% | **23.57%** |


<details><summary>Abstract</summary> 
Although Large Language Models (LLMs) have demonstrated remarkable progress,
their proficiency in graph-related tasks remains notably limited, hindering the
development of truly general-purpose models. Previous attempts, including pretraining graph foundation models or employing supervised fine-tuning, often face
challenges such as the scarcity of large-scale, universally represented graph data.
We introduce G1, a simple yet effective approach demonstrating that Reinforcement
Learning (RL) on synthetic graph-theoretic tasks can significantly scale LLMs’
graph reasoning abilities. To enable RL training, we curate Erdős, the largest
graph reasoning dataset to date comprising 50 diverse graph-theoretic tasks of
varying difficulty levels, 100k training data and 5k test data, all drived from realworld graphs. With RL on Erdős, G1 obtains substantial improvements in graph
reasoning, where our finetuned 3B model even outperforms Qwen2.5-72B-Instruct
(24x size). RL-trained models also show strong zero-shot generalization to unseen tasks, domains, and graph encoding schemes, including other graph-theoretic
benchmarks as well as real-world node classification and link prediction tasks,
without compromising general reasoning abilities. Our findings offer an efficient,
scalable path for building strong graph reasoners by finetuning LLMs with RL on
graph-theoretic tasks, which combines the strengths of pretrained LLM capabilities with abundant, automatically generated synthetic data, suggesting that LLMs
possess graph understanding abilities that RL can elicit successfully.
</details>



## 📌 Key Takeaways
1️⃣ **Erdős: The Largest Graph-Theoretic Benchmark**

We introduce Erdős, the largest and most comprehensive graph-theoretic dataset till now, featuring **50 diverse real-world graph tasks** of varying complexities and **a total of 100,000 samples**. Designed for robust evaluation, Erdős provides a reliable platform for training and benchmarking graph reasoning models.

2️⃣ **First RL-Driven Graph Reasoning Model**

We pioneer the application of reinforcement learning (RL) to enhance LLMs on graph reasoning. Our G1 models achieve up to **46% improvement over baselines on Erdős**, with **the 7B variant matching OpenAI’s o3-mini** and the **3B model surpassing Qwen2.5-72B-Instruct by notable margins**.

3️⃣ **Strong Generalization Without Compromise**

G1 exhibits zero-shot generalization on unseen graph tasks, improving performance on **other graph reasoning benchmarks** (GraphWiz, GraphArena) and **real-world graphs** (Cora, PubMed). Crucially, it preserves **general reasoning ability** (GSM8K, MATH, MMLU-Pro), proving its versatility.


## 🔥 Open-source Collections

Our models and datasets are released in the huggingface collection [PKU-ML/G1](https://huggingface.co/collections/PKU-ML/g1-683d659e992794fc99618cf2). In detail:

### Dataset
We release our dataset *Erdős* on the huggingface at [PKU-ML/Erdos](https://huggingface.co/datasets/PKU-ML/Erdos).

### Models
We release our models G1-series on the huggingface:
- `G1-3B`: [PKU-ML/G1-3B](https://huggingface.co/PKU-ML/G1-3B)
- `G1-7B`: [PKU-ML/G1-7B](https://huggingface.co/PKU-ML/G1-7B)
- `G1-Zero-3B`: [PKU-ML/G1-Zero-3B](https://huggingface.co/PKU-ML/G1-Zero-3B)
- `G1-Zero-7B`: [PKU-ML/G1-Zero-7B](https://huggingface.co/PKU-ML/G1-Zero-7B)
- `G1-Direct-SFT-3B`: [PKU-ML/G1-Direct-SFT-3B](https://huggingface.co/PKU-ML/G1-Direct-SFT-3B)
- `G1-Direct-SFT-7B`: [PKU-ML/G1-Direct-SFT-7B](https://huggingface.co/PKU-ML/G1-Direct-SFT-7B)
- `G1-CoT-SFT-3B`: [PKU-ML/G1-CoT-SFT-3B](https://huggingface.co/PKU-ML/G1-CoT-SFT-3B)
- `G1-Cot-SFT-7B`: [PKU-ML/G1-CoT-SFT-7B](https://huggingface.co/PKU-ML/G1-CoT-SFT-7B)


## 🌟 A Quick Start
We provide an example code for graph reasoning using our G1 models. Run the following:
```bash
python example.py
```

In this example, the task is to detemine the *degree centrality* for a given graph:
> "The task is to determine the degree centrality of a node in the graph. Degree centrality for a node is the fraction of nodes it is connected to.  
> Here is an undirected graph containing nodes from 1 to 15.
> The edges are: (1, 15), (15, 11), (2, 3), (2, 6), (3, 6), (3, 7), (6, 7), (6, 8), (7, 8), (7, 14), (4, 10), (10, 5), (10, 12), (8, 14), (8, 9), (12, 11), (12, 13).  
> Question: What is the degree centrality of node 2 in the graph?
> You need to format your answer as a float number."

The response of G1-7B gives a correct answer of 0.1429 with clear steps:
> "To determine the degree centrality of node 2 in the given undirected graph, we first need to count the number of edges connected to node 2.
> The degree centrality of a node is defined as the fraction of nodes it is connected to, which can be calculated by dividing the number of edges connected to the node by the total number of nodes minus one (since a node does not count itself).  
> Given the edges: - (1, 15) - (15, 11) - (2, 3) - (2, 6) - (3, 6) - (3, 7) - (6, 7) - (6, 8) - (7, 8) - (7, 14) - (4, 10) - (10, 5) - (10, 12) - (8, 14) - (8, 9) - (12, 11) - (12, 13). We identify the edges connected to node 2: - (2, 3) - (2, 6). This gives node 2 a degree of 2, since it is connected by two edges.  
> Next, we calculate the degree centrality by dividing the degree of node 2 by the total number of nodes minus one. The total number of nodes in the graph is 15.  
> Degree Centrality = $\\frac{\\text{Degree of node 2}}{15 - 1} = \\frac{2}{14} = \\frac{1}{7} \\approx 0.1429$. Therefore, the final answer is: $\\boxed{0.1429}$"


## 🚀 Environment Setups
Our implementation is based on the library [verl](https://github.com/volcengine/verl) developed by ByteDance Seed team.

1. Requirements:
- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.1

verl supports various backends. Currently, the following configurations are available: **FSDP** and **Megatron-LM** (optional) for training. **SGLang**, **vLLM** and **TGI** for rollout generation. For more information, please check the documents of [verl](https://verl.readthedocs.io/en/latest/start/install.html).

<!-- 2. For installing the dependencies, we recommend to use a fresh new conda environment:
```bash
conda create -n verl python==3.10
conda activate verl
``` -->

2. For installing the dependencies, I have modified to use a uv environment
```bash
source uv_install.sh
```

<!-- 3. Then, execute the install.sh script that we provided in the repository:
```bash
# Make sure you have activated verl conda env
# If you need to run with megatron
bash install.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash install.sh
``` -->

If you encounter any issues during installation, please refer to the [Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html#) provided by Verl. If problems persist, don’t hesitate to [report them to us](https://github.com/PKU-ML/G1/issues).



## 🎯 Running Supervised Fine-tuning 

### Rejection Sampling (Optional)

For rejection sampling, we randomly sample 100 examples per task from *Erdős*, and use Qwen2.5-32B-Instruct to generate responses for k = 8 times with a temperature of 1.0. We filter the responses with right answers, and resample from *Erdős* with a different random seed and repeat the process above if necessary.  

To run rejection sampling, execute the following command:

``` bash
cd evaluation/Erdos
python rejection_sampling.py \
    --n_sample 8 \
    --model Qwen/Qwen2.5-32B-Instruct \
    --save_path data/rft_data \
    --temperature 1.0 \
    --num_gpu 4 \
    --seed 42
```
- `n_sample`: Number of responses to generate per example (k=8)
- `model`: Path to generation model
- `save_path`: Output directory for valid query-response pairs
- `temperature`: Sampling temperature (fixed at 1.0 for diversity)

**Note**: We have uploaded the rejection sampled dataset to [PKU-ML/Erdos-CoT](https://huggingface.co/datasets/PKU-ML/Erdos-CoT) for direct usage.


### Preprocess SFT data
Transfer the format of data for SFT training:
```bash
python preprocess_sft_data.py \
    --data_source PKU-ML/Erdos-CoT \
    --save_dir datasets/graph_cot_sft \
    --tokenizer_path Qwen/Qwen2.5-3B-Instruct
```
- `data_source`: The original data directory
- `save_dir`: The transferred data directory
- `tokenizer_path`: The path for tokenizer


### Supervised Finetuning

Run supervised finetuing with the following script:
```bash
bash run_sft.sh
```
**Key customizable parameters for the training script:**
- `save_path`: Output directory for the trained model
- `data.train_files` and `data.val_files`: Paths to the processed dataset.
- `model.partial_pretrain`: The path to the base model.
    


## 🧩 Running Reinforcement Learning
Follow these steps to reproduce our G1 implementation:

1. **Download the dataset.**
   
   Download the dataset *Erdős* from [PKU-ML/Erdos](https://huggingface.co/datasets/PKU-ML/Erdos) and save it to locally (*e.g.* data/erdos).
3. **Preprocess the dataset for RL training.**

   Run the preprocessing script to convert the dataset format:
   ```bash
   python preprocess_graph.py --data_source PKU-ML/Erdos  --local_dir data/erdos
   ```
   - `local_dir`: Output directory for the processed dataset.
5. **Launch RL Training.**
   
   Execute the training script (3B model as an example):
   ```bash
   bash run_3B.sh
   ```
 **Configuration notes for the training script:**
 
 1. **Key customizable parameters:**
 - `SAVE_DIR`: Output directory for the trained model.
 - `graph_train_path` and `graph_test_path`: Paths to the processed dataset.
 - Logging: Defaults to [Tensorboard](https://www.tensorflow.org/tensorboard). To use [Weights & Biases](https://wandb.ai/site/), set `trainer.logger = ['console','wandb']`.


 2. **GPU requirements:**
  - Our paper used 8×A800 GPUs. For limited GPU resources, reduce these parameters (may affect performance):
    ```bash
    actor_rollout_ref.actor.ppo_mini_batch_size
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu
    ```
    See [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for more instructions.
 - Set `trainer.n_gpus_per_node` to your actual GPU count.


## 🌊 Inference and Evaluation
Follow these steps to evaluate G1 models and baseline methods:

1. **Convert RL-trained Checkpoints to HuggingFace Format**
   
   Merge the model checkpoints into a HuggingFace-compatible format:
   ```bash
   python model_merger.py \
    --backend fsdp \
    --hf_model_path Qwen/Qwen2.5-3B-Instruct \
    --local_dir  models/G1_3B/global_step_300/actor \
    --target_dir our_models/G1_3B
   ```
   - `hf_model_path`: Path to the original base model.
   - `local_dir`: Directory containing RL-trained checkpoints.
   - `target_dir`: Output directory for the converted model.

2. **Configure Tokenizer Files**

   Copy required tokenizer files from the base model:
   ```bash
   cp models/Qwen2.5-3B-Instruct/tokenizer_config.json our_models/G1_3B/tokenizer_config.json
   cp models/Qwen2.5-3B-Instruct/tokenizer.json our_models/G1_3B/tokenizer.json
   cp models/Qwen2.5-3B-Instruct/vocab.json  our_models/G1_3B/vocab.json
   ```

3. **Run Comprehensive Evaluation**

   We provide evaluation scripts for multiple benchmarks:
   | Benchmark Type	| Included Tests |
   |--------------|------------------|
   | Graph Reasoning | Erdős, [GraphWiz](https://arxiv.org/abs/2402.16029), [GraphArena](https://arxiv.org/abs/2407.00379) |
   | Graph Real-World Tasks |  [Node Classification](https://arxiv.org/abs/2502.18771), [Link Prediction](https://arxiv.org/abs/2502.18771) |
   | General Reasoning | [GSM8K](https://arxiv.org/abs/2110.14168), [MATH](https://arxiv.org/abs/2103.03874), [MMLU-pro](https://arxiv.org/abs/2406.01574) |

   For detailed instructions, see the evaluation [README.md](https://github.com/PKU-ML/G1/blob/main/evaluation/README.md) .
   


## 🎨 Customization Guide
To adapt G1 for your needs, we recommend modifying these key files from our verl-based implementation:
```markdown
trainer/               # Core training components
  - main_ppo.py       # Main training entry point (handles PPO loop)
  - ppo/reward.py     # Reward computation and shaping logic

workers/              # Parallel processing components
  - reward_manager/__init__.py # Reward calculation interface
  - reward_manager/graph.py    # Graph-specific reward functions

utils/                # Supporting utilities
  - reward_score/__init__.py   # Reward normalization/scaling
  - reward_score/graph.py      # Graph-specific scoring metrics

graph_meta_data.json  # Dataset metadata and configuration
```


## Citation
If you find this work useful, please give us a free cite:
```bibtex
@article{guo2025g1,
  title={G1: Teaching LLMs to Reason on Graphs with Reinforcement Learning},
  author={Guo, Xiaojun and Li, Ang and Wang, Yifei and Jegelka, Stefanie and Wang, Yisen},
  journal={arXiv preprint arXiv:2505.18499},
  year={2025}
}
```
