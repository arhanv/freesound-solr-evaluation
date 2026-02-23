
import streamlit as st
import subprocess
import os
import sys
import threading
import queue
import time
import re
import glob
import json
import pickle
from datetime import datetime

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from search.stats_utils import get_similarity_spaces
from search.configs import SOLR_URL
from sidebar_utils import render_sidebar_health

render_sidebar_health()

st.title("Synthetic Data Management")
st.caption(
    "Generate realistic synthetic embeddings (via GMM) to test performance at scale."
)

from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row
import pandas as pd

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("synth_process", None),
    ("synth_log_lines", []),
    ("synth_log_queue", queue.Queue()),
    ("synth_progress_pct", 0),
    ("synth_progress_text", "Starting..."),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_gmm_models_df():
    """Returns a DataFrame of saved GMM models from models/gmm/."""
    model_dir = os.path.join(root_dir, "models", "gmm")
    if not os.path.isdir(model_dir):
        return pd.DataFrame()

    rows = []
    # Match both .json and .json.*.archive
    for meta_path in glob.glob(os.path.join(model_dir, "*.json*")):
        if not (meta_path.endswith(".json") or meta_path.endswith(".archive")):
            continue
        status = "Auto-Archived" if ".archive" in meta_path else "Latest"
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            continue

        model_filename = meta.get("model_filename")
        if not model_filename:
            model_filename = os.path.basename(meta_path).replace(".json", ".pkl")
        filepath = os.path.join(model_dir, model_filename)
        if not os.path.exists(filepath):
            continue

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        ts = meta.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        rows.append({
            "Status": [status],  # List for multiselect pill UI
            "Source Space": meta.get("source_space", "?"),
            "Created": ts,
            "Training Size": meta.get("n_training_samples", meta.get("n_vectors", "?")),
            "K (components)": meta.get("k", "?"),
            "Fingerprint": meta.get("dataset_fingerprint", "N/A"),
            "Cov Type": meta.get("covariance_type", "?"),
            "Size (MB)": f"{size_mb:.1f}",
            "File": model_filename,
            "Path": filepath,
            "BIC Data": meta.get("bic_data"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Created", ascending=False).reset_index(drop=True)
    return df

def highlight_latest(row):
    is_latest = "Latest" in row["Status"]
    color = "rgba(241, 196, 15, 0.15)" if is_latest else ""
    return [f"background-color: {color}" if color else ""] * len(row)


def get_synthetic_stats():
    """Returns (count, space_name) of current synthetic vectors in Solr."""
    try:
        import pysolr
        solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=5)
        count = solr.search("is_synthetic:true", rows=0).hits
        spaces = get_similarity_spaces()
        synth_spaces = [s["name"] for s in spaces if "_synthetic" in s["name"]]
        return count, synth_spaces
    except Exception:
        return 0, []


def enqueue_output(out, q):
    for line in iter(out.readline, ""):
        q.put(line)
    out.close()


def run_command(cmd):
    """Launch cmd in a subprocess, wiring up the log queue."""
    st.session_state.synth_log_lines = []
    st.session_state.synth_progress_pct = 0
    st.session_state.synth_progress_text = "Initializing..."
    st.session_state.synth_log_queue = queue.Queue()

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=root_dir,
        bufsize=1,
        universal_newlines=True,
    )
    st.session_state.synth_process = p
    t = threading.Thread(target=enqueue_output, args=(p.stdout, st.session_state.synth_log_queue))
    t.daemon = True
    t.start()


# ── Live status metrics ────────────────────────────────────────────────────────
synth_count, synth_spaces = get_synthetic_stats()
current_spaces_len = len(synth_spaces)

# Initialize session state keys if they don't exist
if 'prev_synth_count' not in st.session_state:
    st.session_state.prev_synth_count = synth_count
if 'prev_synth_spaces' not in st.session_state:
    st.session_state.prev_synth_spaces = current_spaces_len

# Calculate deltas
delta_count = synth_count - st.session_state.prev_synth_count
delta_spaces = current_spaces_len - st.session_state.prev_synth_spaces

m1, m2 = st.columns(2)

# Use None for delta if it's 0 to keep the UI clean
m1.metric(
    label="Synthetic Docs in Solr", 
    value=f"{synth_count:,}", 
    delta=delta_count if delta_count != 0 else None
)

m2.metric(
    label="Synthetic Spaces", 
    value=current_spaces_len, 
    delta=delta_spaces if delta_spaces != 0 else None
)
st.divider()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_fit, tab_generate, tab_analyze, tab_manage = st.tabs([
    ":material/fit_screen: Fit Model",
    ":material/add_box: Generate & Index",
    ":material/analytics: Analysis & Visualization",
    ":material/cleaning_services: Manage & Cleanup",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 – FIT
# ─────────────────────────────────────────────────────────────
with tab_fit:
    st.markdown(
        "Train a Gaussian Mixture Model (GMM) on existing real vectors from Solr. "
        "The saved model can later be used to sample synthetic embeddings that match the real distribution."
    )

    with stylable_container(
        key="fit_box",
        css_styles="{ border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem; padding: calc(1em - 1px); }",
    ):
        # Source space
        try:
            spaces = get_similarity_spaces()
            source_options = [s["name"] for s in spaces if "_pca" not in s["name"] and "_synthetic" not in s["name"]]
        except Exception:
            source_options = []
        if not source_options:
            source_options = ["laion_clap"]

        r1 = row([3, 1], vertical_align="bottom")
        with r1.container():
            fit_source = st.selectbox("Source Space", source_options, key="fit_source_space")
        with r1.container():
            fit_seed = st.number_input("Random Seed", value=42, key="fit_seed")

        st.write("")
        use_find_k = st.toggle(
            "Auto-select best K via BIC search",
            value=True,
            key="fit_find_k",
            help="Fits many GMMs across a range of K and picks the lowest BIC. Slow but produces a better model. Saves a BIC curve plot alongside the model.",
        )

        if use_find_k:
            r2 = row(3, vertical_align="center")
            with r2.container():
                fit_min_k = st.number_input("Min K", min_value=2, value=100, step=10, key="fit_min_k")
            with r2.container():
                fit_max_k = st.number_input("Max K", min_value=2, value=1000, step=100, key="fit_max_k")
            with r2.container():
                fit_step = st.number_input("Step", min_value=1, value=100, step=10, key="fit_step")
            fit_k = fit_min_k  # starting point; best_k chosen automatically
        else:
            fit_k = st.number_input("K (number of components)", min_value=1, value=64, step=8, key="fit_k",
                                     help="Number of Gaussian mixture components. More = richer distribution, slower fit.")

        r3 = row([2, 2], vertical_align="center")
        with r3.container():
            fit_max_iter = st.number_input("Max EM Iterations", min_value=10, value=100, step=10, key="fit_max_iter")
        with r3.container():
            fit_subsample = st.number_input(
                "Subsample (0 = use all)",
                min_value=0, value=0, step=1000, key="fit_subsample",
                help="Use a random subset of real vectors for fitting. Advisable when testing on larger datasets.",
            )

        if st.button("Fit GMM", type="primary", disabled=st.session_state.synth_process is not None, key="btn_fit"):
            cmd = [
                "python", "-u", "search/generate_and_index_synthetics.py",
                "--fit",
                "--source-space", fit_source,
                "--seed", str(fit_seed),
                "--k", str(fit_k),
                "--max-iter", str(fit_max_iter),
            ]
            if use_find_k:
                cmd += ["--find-k", "--min-k", str(fit_min_k), "--max-k", str(fit_max_k), "--step", str(fit_step)]
            if fit_subsample > 0:
                cmd += ["--subsample", str(fit_subsample)]
            run_command(cmd)
            st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 2 – GENERATE
# ─────────────────────────────────────────────────────────────
with tab_generate:
    st.markdown(
        "Sample vectors from a saved GMM and index them into Solr. "
        "Each run **adds** to existing synthetic data (it does not replace it). "
        "Use the **Manage / Cleanup** tab to wipe existing synthetics first if needed."
    )

    df_models = get_gmm_models_df()

    with stylable_container(
        key="gen_box",
        css_styles="{ border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 0.5rem; padding: calc(1em - 1px); }",
    ):
        if df_models.empty:
            st.warning("No saved GMM models found in `models/gmm/`. Go to **① Fit GMM Model** first.")
        else:
            display_cols = ["Status", "Source Space", "Created", "Training Size", "K (components)", "Fingerprint", "Cov Type"]
            st.caption("Select a model to use:")
            
            styled_df = df_models[display_cols].style.apply(highlight_latest, axis=1)
            ev = st.dataframe(
                styled_df,
                column_config={
                    "Status": st.column_config.MultiselectColumn("Status", options=["Latest", "Auto-Archived"], color="auto")
                },
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                width="stretch",
            )

            selected_model_row = None
            if ev.selection.rows:
                selected_model_row = df_models.iloc[ev.selection.rows[0]]
                st.success(f"Using model: `{selected_model_row['File']}`")

            r_gen = row([2, 2], vertical_align="bottom")
            with r_gen.container():
                gen_count = st.number_input(
                    "Number of Synthetic Vectors",
                    min_value=1, value=100_000, step=50_000, key="gen_count",
                    help="Total synthetic documents to generate and index.",
                )
            with r_gen.container():
                gen_batch = st.number_input(
                    "Indexing Batch Size", min_value=100, max_value=10000, value=2000, step=500, key="gen_batch",
                )

            can_gen = selected_model_row is not None and st.session_state.synth_process is None
            if st.button("Generate & Index", type="primary", disabled=not can_gen, key="btn_gen"):
                cmd = [
                    "python", "-u", "search/generate_and_index_synthetics.py",
                    "--generate", str(gen_count),
                    "--source-space", selected_model_row["Source Space"],
                    "--model-path", selected_model_row["Path"],
                ]
                run_command(cmd)
                st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 3 – ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab_analyze:
    st.markdown("Visualize how well the synthetic distribution matches the real data.")

    df_models_a = get_gmm_models_df()

    # --- BIC Plot ---
    with st.expander("BIC Curve (from find-k run)", expanded=True):
        models_with_bic = df_models_a[df_models_a["BIC Data"].notnull() & (df_models_a["Status"].apply(lambda s: "Latest" in s))]
        
        if not models_with_bic.empty:
            model_options = models_with_bic["File"].tolist()
            selected_bic_model = st.selectbox("Select Model to view BIC trace", options=model_options, key="bic_select")
            
            row = models_with_bic[models_with_bic["File"] == selected_bic_model].iloc[0]
            bic_data = row["BIC Data"]
            
            import plotly.express as px
            import pandas as pd
            
            df_bic = pd.DataFrame({
                "Number of Components (k)": bic_data["k_range"],
                "BIC Score": bic_data["bic_scores"]
            })
            
            fig = px.line(df_bic, x="Number of Components (k)", y="BIC Score", markers=True, 
                          title=f"BIC Score vs. K for {selected_bic_model}")
            
            # Highlight optimal K
            optimal_k = row["K (components)"]
            fig.add_vline(x=optimal_k, line_dash="dash", line_color="green", annotation_text=f"Optimal K={optimal_k}")
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No BIC curve found in current models. Run **Fit GMM Model** with **Auto-find best K** enabled.")

    # --- t-SNE ---
    with st.expander("t-SNE: Real vs Synthetic Distribution", expanded=True):
        st.markdown(
            "Fetches a sample of real and synthetic vectors directly from Solr and runs t-SNE "
            "to visualize how well the GMM captures the real embedding distribution."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            try:
                source_options_a = [s["name"] for s in get_similarity_spaces() if "_pca" not in s["name"] and "_synthetic" not in s["name"]]
            except Exception:
                source_options_a = ["laion_clap"]
            tsne_source = st.selectbox("Real space", source_options_a, key="tsne_source")
        with col_b:
            tsne_limit = st.number_input(
                "Samples per class", min_value=100, value=10_000, step=100, key="tsne_limit",
                help="Number of real AND synthetic vectors to fetch each. Higher = slower but more representative.",
            )
        with col_c:
            tsne_perplexity = st.number_input("t-SNE perplexity", min_value=5, max_value=100, value=30, key="tsne_perplexity")

        if st.button("Run t-SNE", key="btn_tsne", disabled=st.session_state.synth_process is not None):
            with st.spinner("Fetching vectors and running t-SNE (this may take a minute)..."):
                try:
                    import numpy as np
                    import pysolr
                    import plotly.express as px
                    from sklearn.manifold import TSNE

                    solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=30)

                    def fetch_sample(query, limit):
                        results = solr.search(query, rows=limit, fl="sim_vector*", sort="random_1 asc")
                        vecs = []
                        for doc in results.docs:
                            vec_field = next((k for k in doc if k.startswith("sim_vector")), None)
                            if vec_field:
                                vecs.append(doc[vec_field])
                        return np.array(vecs, dtype=np.float32) if vecs else np.empty((0,))

                    real_vecs = fetch_sample(f"content_type:v AND similarity_space:{tsne_source}", tsne_limit)
                    synth_space = f"{tsne_source}_synthetic"
                    synth_vecs = fetch_sample(f"content_type:v AND similarity_space:{synth_space}", tsne_limit)

                    if len(real_vecs) == 0 or len(synth_vecs) == 0:
                        st.error(
                            f"Could not fetch vectors. "
                            f"Real: {len(real_vecs)}, Synthetic: {len(synth_vecs)}. "
                            f"Make sure synthetic data has been generated."
                        )
                    else:
                        X = np.vstack([real_vecs, synth_vecs])
                        labels = ["Real"] * len(real_vecs) + ["Synthetic"] * len(synth_vecs)

                        tsne = TSNE(
                            n_components=2, perplexity=int(tsne_perplexity),
                            max_iter=1000, random_state=42, init="pca", learning_rate="auto",
                        )
                        X_2d = tsne.fit_transform(X)

                        df_tsne = pd.DataFrame({"x": X_2d[:, 0], "y": X_2d[:, 1], "Type": labels})
                        fig = px.scatter(
                            df_tsne, x="x", y="y", color="Type",
                            color_discrete_map={"Real": "#1f77b4", "Synthetic": "#ff7f0e"},
                            opacity=0.45,
                            labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                            title=f"t-SNE: Real vs Synthetic ({len(real_vecs)} + {len(synth_vecs)} vectors)",
                        )
                        fig.update_traces(marker=dict(size=4))
                        fig.update_layout(legend_title_text="")
                        st.plotly_chart(fig, width="stretch")
                        st.caption(
                            f"Real unique vectors: {len(np.unique(real_vecs, axis=0))}/{len(real_vecs)} | "
                            f"Synthetic unique vectors: {len(np.unique(synth_vecs, axis=0))}/{len(synth_vecs)}"
                        )
                except ImportError as e:
                    st.error(f"Missing dependency: {e}. Make sure sklearn is installed.")
                except Exception as e:
                    st.error(f"t-SNE failed: {e}")

    # --- Model Stats ---
    with st.expander("Saved Model Details", expanded=False):
        if df_models_a.empty:
            st.info("No saved GMM models found.")
        else:
            display_cols = ["Status", "Source Space", "Created", "Training Size", "K (components)", "Fingerprint", "Cov Type", "Size (MB)", "File"]
            styled_df = df_models_a[display_cols].style.apply(highlight_latest, axis=1)
            st.dataframe(
                styled_df, 
                column_config={"Status": st.column_config.MultiselectColumn("Status", options=["Latest", "Auto-Archived"], color="auto")},
                hide_index=True, 
                width="stretch"
            )

# ─────────────────────────────────────────────────────────────
# TAB 4 – MANAGE
# ─────────────────────────────────────────────────────────────
with tab_manage:
    st.markdown(f"**{synth_count:,}** synthetic documents currently in Solr.")
    if synth_spaces:
        st.caption("Spaces: " + ", ".join(f"`{s}`" for s in synth_spaces))

    st.divider()
    st.subheader("Manage Spaces")
    if synth_spaces:
        spaces_info = [s for s in get_similarity_spaces() if s["name"] in synth_spaces]
        df_spaces = pd.DataFrame(spaces_info) 
        ev_space = st.dataframe(
            df_spaces[["name", "dimension", "count"]], 
            on_select="rerun", 
            selection_mode="single-row",
            hide_index=True,
            width="stretch"
        )
        
        if ev_space.selection.rows:
            selected_space_name = df_spaces.iloc[ev_space.selection.rows[0]]["name"]
            st.markdown(f"**Selected Space:** `{selected_space_name}`")
            with st.popover(f"Delete Space Vectors", icon=":material/delete:"):
                st.markdown(f"**Delete vectors for `{selected_space_name}`**")
                if "_pca" not in selected_space_name:
                    st.info("Deleting a base synthetic space will NOT delete the dummy parent sounds. Use **Delete All** to clean up parents.")
                if st.button("Delete space vectors", type="primary", key="del_synth_spc"):
                    with st.spinner("Deleting..."):
                        try:
                            from search.generate_and_index_synthetics import delete_synthetic_space
                            delete_synthetic_space(selected_space_name, SOLR_URL)
                            st.toast(f"Vectors for {selected_space_name} deleted!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Cleanup failed: {e}")
    else:
        st.info("No synthetic spaces exist.")

    st.divider()
    st.subheader("Manage Saved Models")
    df_manage = get_gmm_models_df()
    if df_manage.empty:
        st.info("No saved GMM models found in `models/gmm/`.")
    else:
        display_cols = ["Status", "Source Space", "Created", "Training Size", "K (components)", "Fingerprint", "Cov Type", "Size (MB)", "File"]
        styled_df = df_manage[display_cols].style.apply(highlight_latest, axis=1)
        ev_model = st.dataframe(
            styled_df,
            column_config={"Status": st.column_config.MultiselectColumn("Status", options=["Latest", "Auto-Archived"], color="auto")},
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            width="stretch",
        )

        if ev_model.selection.rows:
            selected_model = df_manage.iloc[ev_model.selection.rows[0]]
            st.markdown(f"**Selected Model:** `{selected_model['File']}`")
            with st.popover(f"Delete GMM Model", icon=":material/delete:"):
                st.markdown(f"**Delete `{selected_model['File']}`**")
                if st.button("Delete Model File + Metadata", type="primary", key="del_gmm_model"):
                    try:
                        os.remove(selected_model["Path"])
                        meta_path = selected_model["Path"].replace(".pkl", ".json")
                        # For archived models, Path will end in .pkl.timestamp.archive. 
                        # To find the json, we mirror the glob logic.
                        if "Auto-Archived" in selected_model["Status"]:
                            # Attempt to reconstruct metadata path
                            base = selected_model["Path"].replace(".pkl", ".json")
                            if os.path.exists(base):
                                os.remove(base)
                        else:
                            if os.path.exists(meta_path):
                                os.remove(meta_path)
                                
                        plot_path = selected_model["Path"].split(".pkl")[0] + "_bic_plot.png"
                        if os.path.exists(plot_path):
                            os.remove(plot_path)
                            
                        st.toast("Model files deleted!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete model: {e}")

    st.divider()
    st.subheader("Delete All Synthetic Data")
    st.warning(
        "This deletes **all** documents tagged `is_synthetic:true` and any space matching "
        "`*_synthetic*`. This cannot be undone.",
        icon="⚠️",
    )

    if synth_count == 0:
        st.info("No synthetic data found — nothing to clean up.")
    else:
        with st.popover("Delete Synthetic Data", icon=":material/delete:", width="stretch"):
            st.markdown(f"**You are about to delete {synth_count:,} synthetic documents from Solr.**")
            if st.button("Yes, delete all synthetic data", type="primary", key="btn_cleanup"):
                with st.spinner("Deleting..."):
                    try:
                        from search.generate_and_index_synthetics import cleanup_synthetic
                        cleanup_synthetic(SOLR_URL)
                        st.toast("Synthetic data deleted!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Cleanup failed: {e}")


# ── Progress monitor (shared across all tabs) ─────────────────────────────────
st.divider()
status_label = "PCA in progress..." if st.session_state.synth_process else "Ready."

with st.status(status_label, expanded=bool(st.session_state.synth_process)) as status:
    progress_bar = st.progress(0, text="Initializing...")

    if st.session_state.synth_process:
        process = st.session_state.synth_process

        while not st.session_state.synth_log_queue.empty():
            try:
                line = st.session_state.synth_log_queue.get_nowait()
                st.session_state.synth_log_lines.append(line)

                m = re.search(r"\b(\d{1,3})%\|", line)
                if m:
                    st.session_state.synth_progress_pct = int(m.group(1))
                m2 = re.search(r"^([^:]+):\s+\d{1,3}%\|", line)
                if m2:
                    st.session_state.synth_progress_text = m2.group(1).strip()
            except queue.Empty:
                break

        retcode = process.poll()
        if retcode is None:
            pct = st.session_state.synth_progress_pct
            label = f"{st.session_state.synth_progress_text}: {pct}%"
            status.update(label=label, state="running")
            progress_bar.progress(pct, text=label)
            time.sleep(0.3)
            st.rerun()
        else:
            if retcode == 0:
                status.update(label="Completed! ✅", state="complete", expanded=False)
                progress_bar.progress(100, text="Done.")
            else:
                status.update(label=f"Failed (exit {retcode}) ❌", state="error")
            st.session_state.synth_process = None
            st.session_state.synth_log_queue = queue.Queue()
    else:
        st.write("No operation running.")

# Log expander
full_log = "".join(st.session_state.synth_log_lines)
clean_log = re.sub(r"\x1b\[[0-9;]*m", "", full_log)
with st.expander("View full log", expanded=False):
    st.code(clean_log if clean_log.strip() else "(no output yet)", language="bash", line_numbers=True)
