
import streamlit as st
import subprocess
import os
import sys
import threading
import queue
import time
import re
import glob

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from search.stats_utils import get_similarity_spaces
from sidebar_utils import render_sidebar_health
render_sidebar_health()

st.title("Apply PCA to Similarity Spaces")
st.caption(
    "Fit PCA models on existing Solr vectors and/or re-index them at reduced dimensionality. "
    "Uses `search/pca.py` under the hood."
)

# --- Session State ---
for key, default in [
    ("pca_process", None),
    ("pca_log_lines", []),
    ("pca_log_queue", queue.Queue()),
    ("pca_progress_pct", 0),
    ("pca_progress_text", "Starting..."),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Helpers ---
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row
import pandas as pd
import json
from datetime import datetime


def get_models_df():
    """Returns a DataFrame of existing PCA models from models/pca/."""
    model_dir = os.path.join(root_dir, "models", "pca")
    if not os.path.isdir(model_dir):
        return pd.DataFrame()
    
    rows = []
    for meta_path in glob.glob(os.path.join(model_dir, "*.json*")):
        is_archived = ".archive" in meta_path
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception:
            continue
            
        model_filename = meta.get('model_filename')
        if not model_filename:
            base = os.path.basename(meta_path)
            model_filename = base.split('.json')[0] + '.pkl'
            if is_archived:
                timestamp = base.split('.json.')[1].replace('.archive', '')
                model_filename += f".{timestamp}.archive"

        filepath = os.path.join(model_dir, model_filename)
        if not os.path.exists(filepath):
            continue
            
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        status = "Archived" if is_archived else "Latest"
        
        timestamp_str = meta.get('timestamp', '')
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str)
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        n_comp = meta.get('n_components', 128)
        source_sp = meta.get('source_space', 'Unknown')
        
        fingerprint = meta.get('dataset_fingerprint', 'N/A')
        corpus_size = meta.get('total_corpus_size', 'N/A')
        id_range = meta.get('parent_id_range', {})
        id_min = id_range.get('min', 'N/A') if isinstance(id_range, dict) else 'N/A'
        id_max = id_range.get('max', 'N/A') if isinstance(id_range, dict) else 'N/A'

        rows.append({
            "File": model_filename,
            "Status": status,
            "Source Space": source_sp,
            "Dimensions": n_comp,
            "Training Size": meta.get('n_training_samples', 'Unknown'),
            "Expl. Variance": f"{meta.get('explained_variance', 0):.2%}" if meta.get('explained_variance') else "Unknown",
            "Created": timestamp_str,
            "Size (MB)": f"{size_mb:.1f}",
            "Fingerprint": fingerprint,
            "Corpus Size": corpus_size,
            "ID Range": f"{id_min} - {id_max}",
            "Path": filepath,
            "Target Space": f"{source_sp}_pca{n_comp}"
        })
        
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Status", "Created"], ascending=[False, False]).reset_index(drop=True)
    return df


def config_content():
    st.subheader("Configuration")
    st.space("small")
    # --- Row 1: Mode ---
    mode = st.segmented_control(
        label="Mode",
        options=["Fit :material/arrow_forward: Transform :material/arrow_forward: Reindex", "Apply Saved Model", "Fit and Save Model"],
        key="pca_mode",
        help="""**Fit ➔ Transform ➔ Reindex**: Train model(s), get projections, and update Solr in one go.  
        **Fit and Save Model**: Train and save PCA model(s) without updating Solr (useful for large batches because indexing takes a bit longer).  
        **Apply Saved Model**: Load a saved model and add its projected embeddings to Solr.
        """,
        default="Fit :material/arrow_forward: Transform :material/arrow_forward: Reindex"
    )

    source_space = None
    dims = []
    workers = 1
    batch_size = 1000
    use_cache = False
    model_path = None
    target_space = None

    if mode == "Apply Saved Model":
        df_models = get_models_df()
        if df_models.empty:
            st.warning("No existing PCA models found in `models/pca/`. Run **Fit and Save Model** first.")
        else:
            st.write("Select a model to apply:")
            event = st.dataframe(
                df_models.drop(columns=["Path", "Target Space", "Fingerprint", "Corpus Size", "ID Range"]), 
                on_select="rerun", 
                selection_mode="single-row",
                width='stretch',
                hide_index=True
            )
            
            selected_rows = event.selection.rows
            if selected_rows:
                selected_idx = selected_rows[0]
                selected_model = df_models.iloc[selected_idx]
                target_dim = selected_model["Dimensions"]
                target_space = selected_model["Target Space"]
                model_path = selected_model["Path"]
                source_space = selected_model["Source Space"]
                dims = [target_dim]
                st.success(f"Selected model: `{selected_model['File']}`")
                st.caption(
                    f"**Training Set Fingerprint:** `{selected_model['Fingerprint']}` | "
                    f"**ID Range:** `{selected_model['ID Range']}` | "
                    f"**Corpus Size at Training:** `{selected_model['Corpus Size']}`"
                )
            else:
                st.info("Please select a model from the table above.")

            # --- Row 3: Workers + Batch size ---
            r3 = row([2, 2, 2], vertical_align="center")
            with r3.container():
                workers = st.number_input(
                    "Parallel Workers", min_value=1, max_value=32, value=1, step=1, key="pca_workers",
                    help="Number of parallel threads for batch indexing. Start at 1–4; higher values help when Solr latency dominates."
                )
            with r3.container():
                batch_size = st.number_input(
                    "Batch Size", min_value=1, max_value=1000, value=1000, step=100, key="pca_batch_size",
                    help="Number of parent documents processed per Solr round-trip. Capped at 1000 by Solr Boolean clause limits."
                )
            with r3.container():
                use_cache = st.toggle(
                    "Cache transformed vectors", value=False, key="pca_cache",
                    help="Save PCA-transformed vectors to disk so reindex can be resumed without re-downloading from Solr (in case of crash or exit). They will be deleted after a successful run."
                )
    else:
        # --- Row 2: Source space + Dimensions ---
        r2 = row([3, 3], vertical_align="bottom")

        if mode == "Fit and Save Model":
            st.markdown("""This will train and save a PCA model (.pkl) to `models/pca/`.  
            **Nothing will be added to Solr.** You can apply this model to the embeddings later using the **Apply Saved Model** tab.""")

        with r2.container():
            try:
                spaces = get_similarity_spaces()
                source_options = [s["name"] for s in spaces if "_pca" not in s["name"]]
            except Exception:
                source_options = []
            if not source_options:
                source_options = ["laion_clap"]

            source_space = st.selectbox(
                "Source Space",
                options=source_options,
                index=0,
                key="pca_source_space",
                help="The existing high-dimensional vector space to reduce.",
            )

        with r2.container():
            dims_raw = st.text_input(
                "Target Dimensions",
                value="32 64 128 256",
                key="pca_dims",
                help="Space-separated list of target dimensions, e.g. `64 128 256`.",
            )
            try:
                dims = [int(d) for d in dims_raw.split() if d.isdigit()]
            except Exception:
                dims = [128]

        # Hide workers & batch size configuration if "Fit and Save Model"
        if mode == "Fit :material/arrow_forward: Transform :material/arrow_forward: Reindex":
            # --- Row 3: Workers + Batch size ---
            r3 = row([2, 2, 2], vertical_align="center")

            with r3.container():
                workers = st.number_input(
                    "Parallel Workers",
                    min_value=1,
                    max_value=32,
                    value=1,
                    step=1,
                    key="pca_workers",
                    help="Number of parallel threads for batch indexing. Start at 1–4; higher values help when Solr latency dominates.",
                )

            with r3.container():
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=1000,
                    value=1000,
                    step=100,
                    key="pca_batch_size",
                    help="Number of parent documents processed per Solr round-trip. Capped at 1000 by Solr Boolean clause limits.",
                )

            with r3.container():
                use_cache = st.toggle(
                    "Cache transformed vectors",
                    value=False,
                    key="pca_cache",
                    help="Save PCA-transformed vectors to disk so reindex can be resumed without re-downloading from Solr (in case of crash or exit). They will be deleted after a successful run.",
                )

    return mode, source_space, dims, workers, batch_size, use_cache, model_path, target_space


with stylable_container(
    key="pca_config_box",
    css_styles="""
        {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            padding: calc(1em - 1px);
        }
    """,
):
    mode, source_space, dims, workers, batch_size, use_cache, model_path, target_space = config_content()


# --- Subprocess runner ---
def enqueue_output(out, q):
    for line in iter(out.readline, ""):
        q.put(line)
    out.close()


def run_pca():
    """Builds the pca.py command and launches it in a subprocess."""
    cmd = [
        "python", "-u", "search/pca.py",
        "--dims", *[str(d) for d in dims],
        "--batch-size", str(batch_size),
        "--workers", str(workers),
    ]

    # Source space is only omitted if Apply & Index and using a specific file model without metadata Source Space, which shouldn't happen
    if source_space:
        cmd.extend(["--source-space", source_space])

    if mode in ("Fit and Save Model", "Fit :material/arrow_forward: Transform :material/arrow_forward: Reindex"):
        cmd.append("--fit")
    if mode in ("Apply Saved Model", "Fit :material/arrow_forward: Transform :material/arrow_forward: Reindex"):
        cmd.append("--reindex")
    if use_cache:
        cmd.append("--cache")
        
    if mode == "Apply Saved Model" and model_path and target_space:
        cmd.extend(["--model-path", model_path, "--target-space", target_space])

    st.session_state.pca_log_lines = []
    st.session_state.pca_progress_pct = 0
    st.session_state.pca_progress_text = "Initializing..."
    st.session_state.pca_log_queue = queue.Queue()

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=root_dir,
        bufsize=1,
        universal_newlines=True,
    )
    st.session_state.pca_process = p

    t = threading.Thread(target=enqueue_output, args=(p.stdout, st.session_state.pca_log_queue))
    t.daemon = True
    t.start()


# Validate before allowing start
can_start = bool(dims) and st.session_state.pca_process is None

if not dims:
    st.warning("Enter at least one valid integer dimension above.")

if st.button(
    "Start PCA",
    type="primary",
    disabled=not can_start,
):
    run_pca()
    st.rerun()

st.divider()

# --- Status + live log ---
status_label = "Ready to start..."
if st.session_state.pca_process:
    status_label = "PCA in progress..."

with st.status(status_label, expanded=True) as status:
    progress_bar = st.progress(0, text="Initializing...")
    log_area = st.empty()

    if st.session_state.pca_process:
        process = st.session_state.pca_process

        # Drain the queue
        while not st.session_state.pca_log_queue.empty():
            try:
                line = st.session_state.pca_log_queue.get_nowait()
                st.session_state.pca_log_lines.append(line)

                # Parse tqdm-style progress: "N%|..." or "NN%|..."
                m = re.search(r"\b(\d{1,3})%\|", line)
                if m:
                    st.session_state.pca_progress_pct = int(m.group(1))

                # Pick up the description before the progress bar, e.g. "Downloading vectors: 42%|"
                m2 = re.search(r"^([^:]+):\s+\d{1,3}%\|", line)
                if m2:
                    st.session_state.pca_progress_text = m2.group(1).strip()

            except queue.Empty:
                break

        retcode = process.poll()

        if retcode is None:
            pct = st.session_state.pca_progress_pct
            label = f"{st.session_state.pca_progress_text}: {pct}%"
            status.update(label=label, state="running")
            progress_bar.progress(pct, text=label)
            time.sleep(0.3)
            st.rerun()
        else:
            if retcode == 0:
                status.update(label="PCA Completed! ✅", state="complete", expanded=False)
                progress_bar.progress(100, text="Done.")
            else:
                status.update(label=f"PCA Failed (exit code {retcode}) ❌", state="error")
            st.session_state.pca_process = None
            st.session_state.pca_log_queue = queue.Queue()
    else:
        st.write(
            "Configure the options above and click **Start PCA** to begin. "
            "Progress will stream here in real time."
        )

# --- Full log expander ---
full_log = "".join(st.session_state.pca_log_lines)
clean_log = re.sub(r"\x1b\[[0-9;]*m", "", full_log)  # strip ANSI

with st.expander("View full log", expanded=False):
    st.code(clean_log if clean_log.strip() else "(no output yet)", language="bash", line_numbers=True)
