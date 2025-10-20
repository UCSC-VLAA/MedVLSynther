from pathlib import Path
import json
import click
import pandas as pd
import tqdm

def merge_output(output_dir):
    output_dir = Path(output_dir)

    output_shard_dir = output_dir / "shards"
    if not output_shard_dir.exists():
        raise FileNotFoundError(
            f"Output directory '{output_shard_dir}' does not exist."
        )

    out_file_list = output_shard_dir.glob("*.jsonl")
    results = []
    for out_file_ in tqdm.tqdm(out_file_list):
        with open(out_file_) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    results.append(result)

    out_file = output_dir / "eval_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} records to '{out_file}'.")

    out_acc_file = out_file.parent / f"acc-{out_file.stem}.json"
    result_acc = compute_results_acc(out_file)
    print(f"Accuracy: {result_acc}")
    with open(out_acc_file, "w", encoding="utf-8") as f:
        json.dump(result_acc, f, indent=2, ensure_ascii=False)
    print(f"Saved accuracy to '{out_acc_file}'.")


def compute_results_acc(out_file):
    results = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Loading results"):
            if line.strip():
                result = json.loads(line.strip())
                results.append(
                    {
                        "num_rollouts": result["num_rollouts"],
                        "num_correct": result["num_correct"],
                        "dataset_name": result.get("dataset_name", "default"),
                    }
                )

    df = pd.DataFrame(results)
    df["acc_total"] = df["num_correct"] / df["num_rollouts"]
    df["acc_pass@total"] = df["num_correct"] > 0
    results_acc_total = df.groupby("dataset_name")["acc_total"].mean().to_dict()
    results_acc_pass_at_total = df.groupby("dataset_name")["acc_pass@total"].mean().to_dict()
    results_num_samples = df.groupby("dataset_name").size().to_dict()

    # NOTE(xk) Avoid: TypeError: Object of type int64 is not JSON serializable
    total_num_rollouts = int(df["num_rollouts"].sum())
    total_num_correct = int(df["num_correct"].sum())

    outputs = {
        "accuracy_total": results_acc_total,
        "accuracy_pass@total": results_acc_pass_at_total,
        "results_num_samples_by_dataset_name": results_num_samples,
        "total_num_rollouts": total_num_rollouts,
        "total_num_correct": total_num_correct,
        "num_samples": len(df),
    }

    return outputs

@click.command()
@click.option("--result_dir", "-d", type=click.Path(exists=True), default="outputs", help="Output directory containing the shards.")
@click.option("--skip_merge", is_flag=True, help="Skip merging shards and only compute accuracy.")
def main(result_dir, skip_merge):
    """Merge output shards and compute accuracy."""
    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory '{result_dir}' does not exist.")

    if not skip_merge:
        merge_output(result_dir)

    output_dir = result_dir
    out_file = output_dir / "eval_results.jsonl"
    out_acc_file = out_file.parent / f"acc-{out_file.stem}.json"
    result_acc = compute_results_acc(out_file)
    print(f"Accuracy: {result_acc}")
    with open(out_acc_file, "w", encoding="utf-8") as f:
        json.dump(result_acc, f, indent=2, ensure_ascii=False)
    print(f"Saved accuracy to '{out_acc_file}'.")

if __name__ == "__main__":
    main()