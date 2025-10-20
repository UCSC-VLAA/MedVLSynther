python eval/qwen_medical_benchmark.py \
    --model /home/efs/nwang60/models/Qwen2.5-VL-72B-Instruct \
    --dp_size 2 \
    --tp_size 4 \
    --temperature 0.0 \
    --batch_size 32 \
    --output_dir outputs_eval/qwen_medical_benchmark \
    --dataset_name /home/efs/nwang60/datasets/MedVLThinker-Eval \
    --max_tokens 4096