#!/usr/bin/env bash
set -e

MODEL=/home/efs/nwang60/models/Qwen2.5-VL-7B-Instruct
ENGINE=vllm

# —— 载入 .env（WANDB/HF_TOKEN 等） ——
set -a && source .env && set +a

# —— 本地临时盘与缓存：走 /tmp，避开 EFS+mmap ——
mkdir -p /tmp/hf_cache_a /tmp/hf_datasets_a /tmp/ray_tmp_a /tmp/ray_spill_a /tmp/trainset_a

export HF_HOME=/tmp/hf_cache_a
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=/tmp/hf_datasets_a
export HF_DATASETS_DISABLE_MEMORY_MAPPING=1
export ARROW_USE_MMAP=0

export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/ray_tmp_a
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/tmp/ray_spill_a"}}'

# —— 把 parquet 拷到本地盘（若源数据更新请重拷） ——
rsync -a /home/efs/nwang60/datasets/biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset2k_verl/train.parquet /tmp/trainset_a/
rsync -a /home/efs/nwang60/datasets/MedVLThinker-Eval_verl/test.parquet   /tmp/trainset_a/

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/tmp/trainset_a/train.parquet \
  data.val_files=/tmp/trainset_a/test.parquet \
  data.shuffle=True \
  data.train_batch_size=48 \
  data.max_prompt_length=80000 \
  data.filter_overlong_prompts=False \
  data.max_response_length=4096 \
  data.truncation='error' \
  data.image_key=images \
  data.dataloader_num_workers=0 \
  data.val_batch_size=256 \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=24 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='Medvlthinker_synthesis' \
  trainer.experiment_name='qwen2_5_vl_7b_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_subset2k' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=0 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.max_critic_ckpt_to_keep=3 \
  custom_reward_function.path=./train/my_reward.py "$@"