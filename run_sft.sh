#!/bin/bash

export TENSORBOARD_DIR="train_sft_result/graph-cot-sft-3B"
save_path=models/CoT-SFT-3B

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m trainer.fsdp_sft_trainer \
    data.train_files=datasets/Erdos-CoT/data/train.parquet \
    data.val_files=datasets/Erdos-CoT/data/test.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.prompt_dict_keys=["content"] \
    data.response_dict_keys=["content"] \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=3072 \
    data.truncation=left \
    model.partial_pretrain=models/Qwen2.5-3B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=graph-cot-sft-3B \
    trainer.experiment_name=graph-cot-sft-3B \
    trainer.total_epochs=2 \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null
