
import streamlit as st
import subprocess
import os
import sys
import threading
import queue
import time
import re

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from search.stats_utils import get_similarity_spaces
from sidebar_utils import render_sidebar_health

st.set_page_config(page_title="Run Evaluation", layout="wide")

render_sidebar_health()

st.title("Run Evaluation Queries")

# --- Session State Initialization ---
if "eval_process" not in st.session_state:
    st.session_state.eval_process = None
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if "eval_current_phase" not in st.session_state:
    st.session_state.eval_current_phase = "Running..."
if "eval_current_pct" not in st.session_state:
    st.session_state.eval_current_pct = 0
if "eval_batch_pct" not in st.session_state:
    st.session_state.eval_batch_pct = 0
if "eval_batch_text" not in st.session_state:
    st.session_state.eval_batch_text = "Starting..."

# --- Configuration Section ---
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row

def config_content():
    st.subheader("Configuration")
    st.space("small")
    # Row 1: Source Space, Select All Toggle, Dimensions
    r1 = row([3, 2.4, 3], vertical_align="bottom")
    
    with r1.container():
        # Fetch available spaces dynamically
        try:
            spaces = get_similarity_spaces()
        except Exception:
            spaces = []

        # Filter out PCA children
        source_options = [s['name'] for s in spaces if "_pca" not in s['name']]
        if not source_options:
             source_options = ["laion_clap"]

        space = st.selectbox(
            "Source Space", 
            options=source_options, 
            index=0,
            key="eval_source_space",
            help="Select the vector space to evaluate against."
        )

    with r1.container():
        # Toggle for Select All
        all_dims = st.toggle("All available dims", value=True, key="eval_all_dims", help="Automatically select all available PCA dimensions.")

    with r1.container():
        # Dynamic Dimensions Logic
        available_dims = []
        for s in spaces:
            if s['name'].startswith(f"{space}_pca"):
                m = re.search(r'_pca(\d+)', s['name'])
                if m:
                    available_dims.append(int(m.group(1)))
        
        available_dims.sort()
        
        dims_options = available_dims.copy()
        
        if not dims_options:
            st.info(f"No PCA-reduced spaces found for **{space}**.")
            dims = []
        elif all_dims:
            st.multiselect("PCA Dimensions", options=dims_options, default=dims_options, disabled=True, key="eval_dims_multiselect_all")
            dims = dims_options
        else:
            dims = st.multiselect(
                "PCA Dimensions", 
                options=dims_options, 
                default=[],
                key="eval_dims_multiselect",
                help="Select PCA dimensions to test."
            )

    # Row 2: Queries, Warmup, Seed
    r2 = row(3, vertical_align="center")
    
    with r2.container():
        num_sounds = st.number_input(
            "Number of Queries", 
            min_value=1, 
            max_value=100000, 
            value=2000, 
            step=100,
            key="eval_num_sounds",
            help="Number of query sounds to use for evaluation."
        )
        
    with r2.container():
        warmup = st.number_input(
            "Warmup Queries", 
            min_value=0, 
            max_value=5000, 
            value=500, 
            step=50,
            key="eval_warmup",
            help="Number of queries to run before measuring latency."
        )

    # Row 3: Retrieve N, Metric K, Seed
    r3 = row(3, vertical_align="center")
    
    with r3.container():
        retrieve_n = st.number_input(
            "# Neighbors for Retrieval", 
            min_value=1, 
            max_value=1000, 
            value=50, 
            key="eval_retrieve_n",
            help="Number of candidates to retrieve from Solr per query."
        )

    with r3.container():
        metric_k = st.number_input(
            "Top-K Value for Recall Metrics", 
            min_value=1, 
            max_value=retrieve_n, 
            value=min(50, retrieve_n), 
            key="eval_metric_k",
            help="Number of results used to calculate recall metrics. For example, if Top-K=10, Recall@10 represents the percentage of queries for which the correct sound is found within the top 10 retrieved results."
        )

    with r3.container():
        seed = st.number_input("Random Seed", value=42, key="eval_seed", help="Seed for reproducibility.")

    # Row 4: Save Details
    st.write("") # Spacer
    save_details = st.toggle(
        "Save Query-by-Query Metrics", 
        value=True, 
        key="eval_save_details",
        help="If checked, saves individual latency and recall metrics for each query. It may be noticably slower depending on the number of queries, but is recommended for detailed analysis. You can still see global/aggregated metrics without it."
    )
    
    return space, dims, num_sounds, warmup, seed, retrieve_n, metric_k, save_details

with stylable_container(
    key="config_box",
    css_styles="""
        {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            padding: calc(1em - 1px);
        }
    """,
):
    space, dims, num_sounds, warmup, seed, retrieve_n, metric_k, save_details = config_content()


# --- Real-time Log Monitoring ---
def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

def run_evaluation():
    """Starts the evaluation script in a subprocess."""
    
    # Construct command (ensure unbuffered output)
    cmd = [
        "python", "-u", "eval/run_batch_eval.py",
        "--source-space", space,
        "--num-sounds", str(num_sounds),
        "--dims", *[str(d) for d in dims],
        "--warmup", str(warmup),
        "--retrieve-n", str(retrieve_n),
        "--metric-k", str(metric_k),
        "--seed", str(seed),
    ]
    
    if save_details:
        cmd.append("--save-details")
    
    # Always use dashboard mode for the UI
    cmd.append("--dashboard")
    
    st.session_state.log_lines = [] 
    st.session_state.eval_current_phase = "Initializing..."
    st.session_state.eval_current_pct = 0
    st.session_state.eval_batch_pct = 0
    st.session_state.eval_batch_text = "Starting..."
    # Re-create queue to ensure it's clean (though thread safety might be tricky if old thread still writing, but subprocess is new)
    st.session_state.log_queue = queue.Queue()
    
    # Start process
    p = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        cwd=root_dir,
        bufsize=1,
        universal_newlines=True
    )
    st.session_state.eval_process = p
    
    # Start reader thread
    t = threading.Thread(target=enqueue_output, args=(p.stdout, st.session_state.log_queue))
    t.daemon = True
    t.start()

# Start Button
if st.button("Start Evaluation", type="primary", disabled=st.session_state.eval_process is not None):
    run_evaluation()
    st.rerun()

st.divider()
# Container for execution status
status_label = "Ready to start..."
if st.session_state.eval_process:
    status_label = "Evaluation in progress..."

with st.status(status_label, expanded=True) as status:
    # 1. Overall Batch Progress
    overall_progress = st.progress(0, text="Overall Progress")
    # 2. Segment/Phase Progress
    progress_bar = st.progress(0, text="Initializing...")
    log_area = st.empty()
    
    if st.session_state.eval_process:
        process = st.session_state.eval_process
        
        # Consume queue
        while not st.session_state.log_queue.empty():
            try:
                line = st.session_state.log_queue.get_nowait()
                st.session_state.log_lines.append(line)
                
                # Stateful Parsing: Update state as lines arrive
                if "[PHASE]" in line:
                    st.session_state.eval_current_phase = line.replace("[PHASE]", "").strip()
                
                if "[BATCH_PROGRESS]" in line:
                    match = re.search(r'\[BATCH_PROGRESS\]\s+(\d+)/(\d+)', line)
                    if match:
                        curr, total = int(match.group(1)), int(match.group(2))
                        st.session_state.eval_batch_pct = int((curr - 1) / total * 100)
                        st.session_state.eval_batch_text = f"Evaluating Space {curr} of {total}"
                
                match = re.search(r'\[PROGRESS\]\s+(\d+)', line)
                if match:
                    st.session_state.eval_current_pct = int(match.group(1))

            except queue.Empty:
                break
                
        # Check process status
        retcode = process.poll()
        
        if retcode is None:
            # Running
            status.update(label=f"Current: {st.session_state.eval_current_phase}", state="running")
            overall_progress.progress(st.session_state.eval_batch_pct, text=st.session_state.eval_batch_text)
            progress_bar.progress(st.session_state.eval_current_pct, text=f"{st.session_state.eval_current_phase}: {st.session_state.eval_current_pct}%")
            
            # Throttled refresh
            time.sleep(0.3)
            st.rerun()
        else:
            # Completed
            status.update(label="Batch Evaluation Completed! ✅", state="complete", expanded=False)
            overall_progress.progress(100, text="All evaluatons finished.")
            progress_bar.progress(100, text="Done.")
            st.session_state.eval_process = None
            st.session_state.log_queue = queue.Queue() 
    else:
        st.write("Click 'Start Evaluation' above to begin.")

# Display Logs (Filtered and tucked in expander)
full_log = "".join(st.session_state.log_lines)
# Remove ANSI escape codes
clean_log = re.sub(r'\x1b\[[0-9;]*m', '', full_log) 
# Filter out dashboard-specific tags
filtered_lines = [
    line for line in clean_log.splitlines() 
    if not any(tag in line for tag in ["[PROGRESS]", "[PHASE]", "[BATCH_PROGRESS]"])
]
filtered_log = "\n".join(filtered_lines)

with st.expander("View full log", expanded=False):
    st.code(filtered_log, language="bash", line_numbers=True)
