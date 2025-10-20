python eval/glm_medical_benchmark.py \
  --model /home/efs/nwang60/models/GLM-4.5V \
  --dataset_name /home/efs/nwang60/datasets/MedVLThinker-Eval \
  --output_dir outputs_eval/glm_medical_benchmark_v2 \
  --tp_size 4 --dp_size 2 \
  --batch_size 16 --max_tokens 2048