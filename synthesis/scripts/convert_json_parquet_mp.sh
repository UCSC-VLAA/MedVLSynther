python synthesis/convert_json_parquet_mp.py \
    --base ~/efs/nwang60/datasets/biomedica_webdataset_24M \
    --input /opt/dlami/nvme/nwang60/datasets/biomedica_webdataset_json_25k/biomedica_webdataset_25k.json \
    --out /opt/dlami/nvme/nwang60/datasets/biomedica_webdataset_parquet_25k \
    --rows-per-shard 10000 \
    --workers 32