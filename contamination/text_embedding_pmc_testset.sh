python text_dedup_and_similarity_report.py \
  --train "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-pmc_vqa/data/*.parquet" \
  --test  "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-Eval/data/*.parquet" \
  --fields "question,options" \
  --run_embed \
  --model "BAAI/bge-m3" \
  --device cuda \
  --out_dir text_embedding_pmc_testset