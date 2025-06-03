#!/bin/bash
export TENSORBOARD_DIR="visual/G1_3B"

SAVE_DIR="models/G1_3B"
graph_train_path="data/graph_task/train.parquet"
graph_test_path="data/graph_task/test.parquet"

train_files="['$graph_train_path']"
test_files="['$graph_test_path']"


python3 -m trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.prompt_key=prompt \
    data.truncation='error' \
    actor_rollout_ref.model.path=models/cot_sft_3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=graph \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='G1_3B' \
    trainer.experiment_name='G1_3B' \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.val_before_train=False \
    trainer.test_freq=25 \
    trainer.n_gpus_per_node=8 \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.total_epochs=3
