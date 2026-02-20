
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

st.set_page_config(page_title="Run Evaluation", layout="centered")

st.title("Run Evaluation Queries")

# --- Session State Initialization ---
if "eval_process" not in st.session_state:
    st.session_state.eval_process = None
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()

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
            help="Select the vector space to evaluate against."
        )

    with r1.container():
        # Toggle for Select All
        all_dims = st.toggle("All available dims", value=True, help="Automatically select all available PCA dimensions.")

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
            st.multiselect("PCA Dimensions", options=dims_options, default=dims_options, disabled=True)
            dims = dims_options
        else:
            dims = st.multiselect(
                "PCA Dimensions", 
                options=dims_options, 
                default=[],
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
            help="Number of query sounds to use for evaluation."
        )
        
    with r2.container():
        warmup = st.number_input(
            "Warmup Queries", 
            min_value=0, 
            max_value=5000, 
            value=500, 
            step=50,
            help="Number of queries to run before measuring latency."
        )

    with r2.container():
        seed = st.number_input("Random Seed", value=42, help="Seed for reproducibility.")

    # Row 3: Save Details
    st.write("") # Spacer
    save_details = st.toggle(
        "Save Per-Query Metrics", 
        value=True, 
        help="If checked, saves detailed latency histograms and per-query stats for analysis."
    )
    
    return space, dims, num_sounds, warmup, seed, save_details

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
    space, dims, num_sounds, warmup, seed, save_details = config_content()


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
        "--seed", str(seed),
    ]
    
    if save_details:
        cmd.append("--save-details")
    
    st.session_state.log_lines = [] 
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
st.subheader("Execution Progress")

# Containers for updates
progress_bar = st.progress(0, text="Ready to start...")
log_container = st.empty()

if st.session_state.eval_process:
    process = st.session_state.eval_process
    
    # Consume queue
    while not st.session_state.log_queue.empty():
        try:
            line = st.session_state.log_queue.get_nowait()
            st.session_state.log_lines.append(line)
        except queue.Empty:
            break
            
    # Check process status
    retcode = process.poll()
    
    if retcode is None:
        # Running
        # Try to parse progress from last few lines
        latest_logs = "".join(st.session_state.log_lines[-5:]) # Check recent lines
        
        # Look for TQDM pattern:  10%|█ |
        # Or "Running Solr queries:  15%"
        match = re.search(r'(\d+)%', latest_logs)
        if match:
            try:
                pct = int(match.group(1))
                progress_bar.progress(pct, text="Evaluation in progress...")
            except:
                pass
        else:
             progress_bar.progress(0, text="Evaluation running... (Check logs below)")

        time.sleep(0.5)
        st.rerun()
        
    else:
        # Completed
        progress_bar.progress(100, text="Evaluation Completed! ✅")
        st.session_state.eval_process = None
        st.session_state.log_queue = queue.Queue() # garbage collect

# Display Logs
full_log = "".join(st.session_state.log_lines)
clean_log = re.sub(r'\x1b\[[0-9;]*m', '', full_log) 
st.code(clean_log, language="bash", line_numbers=True)
