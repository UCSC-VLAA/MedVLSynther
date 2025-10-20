"""
streamlit run analysis/result_viewer.py
"""

import dotenv

dotenv.load_dotenv(override=True)


import json
import math
from pathlib import Path

import datasets
import streamlit as st
from PIL import Image

###############################################################################
# Streamlit page configuration
###############################################################################
st.set_page_config(page_title="Med‑VLM Eval Results Viewer", layout="wide")

###############################################################################
# Helper functions (cached where appropriate)
###############################################################################


@st.cache_data(show_spinner=False, hash_funcs={Path: str})
def load_dataset(name: str, split: str):
    """Load a HuggingFace dataset split and return the Dataset object."""
    return datasets.load_dataset(name, split=split)


@st.cache_data(show_spinner=False, hash_funcs={Path: str})
def load_results(result_jsonl_path: Path):
    """Load model predictions saved as newline‑delimited JSON and return a list."""
    results = []
    if not result_jsonl_path.exists():
        st.warning(
            f"Result file not found at {result_jsonl_path}. Please check the path."
        )
        return results
    with result_jsonl_path.open("r", encoding="utf‑8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def build_result_index(results):
    """Return a mapping from dataset_index to its corresponding result (first match)."""
    return {res["dataset_index"]: res for res in results}


###############################################################################
# Sidebar – dataset / results parameters & controls
###############################################################################

st.sidebar.header("⚙️ Parameters")


def sidebar_text_input(label, key, value):
    """Wrapper that stores the value in session_state so it survives reruns."""
    if key not in st.session_state:
        st.session_state[key] = value
    return st.sidebar.text_input(label, st.session_state[key], key=key)


# Text inputs for paths and names
result_path_str = sidebar_text_input(
    "Result JSONL path",
    "result_path",
    "/opt/dlami/nvme/nwang60/datasets/MVT-synthesis/qwen2_5_vl_7b_biomedica_webdataset_glm_generated_qwen_verified_VQA_parquet_25k_filtered_13k_eval_v6/regraded_eval_results.jsonl",
)

dataset_name = sidebar_text_input(
    "Dataset name", "dataset_name", "/opt/dlami/nvme/nwang60/datasets/MedVLThinker-Eval"
)

dataset_split = sidebar_text_input("Dataset split", "dataset_split", "test")

# Load button
if st.sidebar.button("📥 Load data") or (
    "dataset" not in st.session_state and Path(result_path_str).exists()
):
    with st.spinner("Loading dataset and results..."):
        st.session_state.dataset = load_dataset(dataset_name, dataset_split)
        st.session_state.results = load_results(Path(result_path_str))
        st.session_state.result_index = build_result_index(st.session_state.results)
        # Build list of all dataset_name values for quick filter hints
        st.session_state.all_ds_names = sorted(
            set(st.session_state.dataset["dataset_name"])
            if len(st.session_state.dataset) > 0
            else []
        )
        # By default no filtering: show every index
        st.session_state.filtered_indices = list(range(len(st.session_state.dataset)))

# Early exit if nothing is loaded yet
if "dataset" not in st.session_state:
    st.info("⬅️ Use the sidebar to load a dataset and result file.")
    st.stop()

###############################################################################
# Sidebar – display / filter controls (enabled after loading)
###############################################################################

st.sidebar.header("🔍 Filters & View")

# Show list of available dataset_name values
st.sidebar.text_area(
    "Distinct dataset_name values (for reference)",
    "\n".join(st.session_state.all_ds_names) or "<empty>",
    height=150,
    disabled=True,
)

# dataset_name filter
filter_name = st.sidebar.text_input("Filter by dataset_name (exact match)")
if st.sidebar.button("Apply dataset_name filter"):
    if filter_name:
        st.session_state.filtered_indices = [
            i
            for i in range(len(st.session_state.dataset))
            if st.session_state.dataset[i]["dataset_name"] == filter_name
        ]
    else:
        # Reset filter
        st.session_state.filtered_indices = list(range(len(st.session_state.dataset)))

# Rows per page selection
rows_per_page = st.sidebar.slider(
    "Rows per page", min_value=1, max_value=50, value=10, step=1
)

# Total pages calculation
total_pages = max(1, math.ceil(len(st.session_state.filtered_indices) / rows_per_page))

# Page number selection
page_number = st.sidebar.number_input(
    "Page number", min_value=1, max_value=total_pages, step=1, value=1, format="%d"
)

###############################################################################
# Main page – viewer
###############################################################################

start_idx = (page_number - 1) * rows_per_page
end_idx = start_idx + rows_per_page
current_indices = st.session_state.filtered_indices[start_idx:end_idx]

st.markdown(
    f"### Displaying dataset indices {start_idx}‑{min(end_idx - 1, len(st.session_state.filtered_indices)-1)} "
    f"(page {page_number} / {total_pages})"
)

for dp_idx in current_indices:
    data_point = st.session_state.dataset[dp_idx]
    result = st.session_state.result_index.get(dp_idx, None)
    parsed = result["parsed_outputs"][0] if result else {}

    # Container per example for cleaner layout
    with st.container():
        # Two‑column layout: images left, text right
        img_col, txt_col = st.columns([1, 2])

        # ——————————— Image(s) ————————————
        with img_col:
            images = data_point["images"]
            if isinstance(images, list):
                for img in images:
                    # dataset returns PIL Image already; ensure correct type then display
                    if isinstance(img, Image.Image):
                        st.image(
                            img, use_container_width=True
                        )  # The use_column_width parameter has been deprecated, use use_container_width instead
            else:
                st.write("[No images found]")

        # ——————————— Textual information ————————————
        with txt_col:
            st.markdown(f"**Dataset index:** {dp_idx}")
            st.markdown(f"**Question:** {data_point['question']}")
            options_json = json.loads(data_point["options"])
            # Show options nicely
            st.markdown("**Options:**")
            for key, text in options_json.items():
                st.markdown(f"‐ **{key}.** {text}")

            st.markdown(
                f"**Ground‑truth answer:** {data_point['answer_label']} — {data_point['answer']}"
            )
            st.markdown(f"**dataset_name:** `{data_point['dataset_name']}`")
            st.markdown("---")

            if result:
                st.markdown("#### Model prediction")
                st.code(parsed.get("output_text", ""), language="text")
                st.markdown(f"**Predicted letter:** {parsed.get('pred_letter', 'N/A')}")
                is_correct = parsed.get("is_correct", False)
                st.markdown(f"**Is correct?** {'✅' if is_correct else '❌'}")
            else:
                st.warning("No prediction found for this index.")

        # Visual separator between rows
        st.divider()

###############################################################################
# Footer – small note
###############################################################################

st.caption(
    "Med‑VLM evaluation viewer • Built with Streamlit — reload the page or change the sidebar settings to update the view."
)

