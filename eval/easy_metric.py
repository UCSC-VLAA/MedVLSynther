import json
from pathlib import Path
import pandas as pd
import click
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def process_single_file(file_path: Path):
    model_name = file_path.parent.name
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                records.append({
                    "dataset_name": data.get("dataset_name"),
                    "average_num_correct": data.get("average_num_correct", 0.0),
                })
            except json.JSONDecodeError:
                print(f"Warning：skip {file_path} line with wrong format")

    if not records:
        return None

    df = pd.DataFrame(records)
    df_mean = df.groupby("dataset_name").mean().reset_index()
    df_mean["model_name"] = model_name
    return df_mean


@click.command()
@click.option(
    "--path", "-p",
    default="outputs/greedy",
    help="'regraded_eval_results.jsonl' file root path",
    type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def main(path: Path):
    print(f"Scanning result file in: {path.resolve()}...\n")

    result_files = list(path.glob("**/regraded_eval_results.jsonl"))

    if not result_files:
        print(f"Error: no 'regraded_eval_results.jsonl' in {path}.")
        return

    print(f"Found {len(result_files)} to process.")

    all_dfs = []
    for file in result_files:
        df = process_single_file(file)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("Error: cannot process any result file.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all['average_num_correct'] *= 100

    pivot_table = df_all.pivot_table(
        index="model_name",
        columns="dataset_name",
        values="average_num_correct",
        aggfunc="mean"
    )

    pivot_table['Overall_Average'] = pivot_table.mean(axis=1)

    print("\n--- average_num_correct % ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(pivot_table.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()