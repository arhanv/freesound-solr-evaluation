
import streamlit as st
import pandas as pd
import glob
import os
import sys
import json
import re
import math
import plotly.express as px
from pygwalker.api.streamlit import StreamlitRenderer

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sidebar_utils import render_sidebar_health

st.set_page_config(page_title="Analysis", layout="wide")

render_sidebar_health()

st.title("Analysis & Visualization")

# --- Helper Functions ---
def format_run_label(run_folder_name):
    """Parses run folder name into a readable label."""
    try:
        parts = run_folder_name.split('_')
        ts_str = parts[-1] 
        from datetime import datetime
        dt = datetime.strptime(ts_str, "%Y%m%d-%H%M")
        nice_date = dt.strftime("%d %b %Y | %H:%M")
        
        seed = "Unknown"
        queries = "Unknown"
        for p in parts:
            if p.startswith("seed"):
                seed = p.replace("seed", "")
            if p.startswith("queries"):
                queries = f"{int(p.replace('queries', '')):,}"
        
        return f"{nice_date} ({queries} queries, seed={seed})"
    except Exception:
        return run_folder_name

def get_run_folders():
    """Finds all run folders in eval/results/."""
    results_dir = os.path.join(root_dir, "eval", "results")
    if not os.path.exists(results_dir):
        return []
        
    folders = []
    for f in os.listdir(results_dir):
        full_path = os.path.join(results_dir, f)
        if os.path.isdir(full_path) and not f.startswith(".") and not f.startswith("archived"):
            folders.append(f)
    
    # Sort by actual datetime suffix (newest first)
    def sort_key(f):
        try:
            ts_str = f.split('_')[-1]
            return ts_str # YYYYMMDD-HHMM sorts correctly as string
        except Exception:
            return "00000000-0000"

    folders.sort(key=sort_key, reverse=True)
    return folders

def load_run_data(run_id):
    """Loads config.json, results.csv, and per-query details for a run."""
    run_dir = os.path.join(root_dir, "eval", "results", run_id)
    
    config = {}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            
    results_df = None
    results_path = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        if 'dims' not in results_df.columns and 'similarity_space' in results_df.columns:
            def extract_dim(s):
                m = re.search(r'pca(\d+)', s)
                return int(m.group(1)) if m else 512
            results_df['dims'] = results_df['similarity_space'].apply(extract_dim)
            results_df = results_df.sort_values('dims')

    details_df = None
    detail_files = glob.glob(os.path.join(run_dir, "per_query_details_*.pkl"))
    if detail_files:
        all_details = []
        for f in detail_files:
            try:
                df = pd.read_pickle(f)
                filename = os.path.basename(f)
                if "pca" in filename:
                     m = re.search(r'pca(\d+)', filename)
                     dim = m.group(1) if m else "?"
                     label = f"PCA-{dim}"
                else:
                     label = "Original (512)"
                df['label'] = label
                all_details.append(df)
            except Exception as e:
                st.error(f"Error loading {f}: {e}")
        
        if all_details:
             details_df = pd.concat(all_details, ignore_index=True)

    return config, results_df, details_df

a1, a2 = st.columns(2)
with a1:
    st.markdown("""
    This page contains some widgets to interpret and visualize evaluation results:  
    **Summary Table**: Overview of aggregate metrics to understand general performance across similarity spaces.  
    **Visualizatons**: Plots for various latency and recall metrics.  
    **PyGWalker**: Interactive visualizer to generate additional plots from the data.
    """)
with a2:
    # --- Run Selection ---
    run_folders = get_run_folders()

    if not run_folders:
        st.warning("No evaluation runs found in `eval/results/`.")
        st.stop()

    st.write("Choose the results folder to analyze (default should be most recent):")
    selected_run = st.selectbox(
        "Select Run", 
        options=run_folders, 
        index=0,
        format_func=format_run_label
    )

    config, results_df, details_df = load_run_data(selected_run)

# --- Shared Color Map ---
if results_df is not None:
    # Get labels for color mapping
    if details_df is not None:
        labels = sorted(details_df['label'].unique(), key=lambda x: int(x.split('-')[1]) if '-' in x and 'Original' not in x else 999)
        color_map = {l: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, l in enumerate(labels)}
        
        # Cross-map for global charts (dim -> label -> color)
        dim_to_color = {}
        for label in labels:
            m = re.search(r'(\d+)', label)
            d = str(int(m.group(1)) if m else 512)
            dim_to_color[d] = color_map[label]
    else:
        color_map = {}
        dim_to_color = {}

# --- Metadata Stats ---
seed = config.get('seed', 'Unknown')
num_sounds = config.get('num_sounds', 0)
similarity_space = config.get('source_space', 'Unknown')
index_size_str = f"{config.get('index_size_mb', 0)} MB" if 'index_size_mb' in config else f"{config.get('index_size', 0):,} Docs"
warmup_cutoff = config.get('warmup', 0)
n_neighbors = results_df['n_neighbors'].iloc[0] if 'n_neighbors' in results_df.columns else 0
# Handle retrieve_n / metric_k (with fallbacks for older runs)
retrieve_n = results_df['retrieve_n'].iloc[0] if 'retrieve_n' in results_df.columns else results_df['n_neighbors'].iloc[0] if 'n_neighbors' in results_df.columns else 50
metric_k = results_df['metric_k'].iloc[0] if 'metric_k' in results_df.columns else retrieve_n

st.space("small")
st.write(f"**Source Space**: {similarity_space}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Seed", seed)
m2.metric("n_neighbors", retrieve_n)
m3.metric("Test Queries", f"{num_sounds:,}")
m4.metric("Warmup Queries", f"{warmup_cutoff:,}")
m5.metric("Total Index Size", index_size_str)

st.subheader("Summary")
st.dataframe(results_df[["dims", "mean_latency", "recall_mean", "ndcg_mean", "qps", "space_size_mb"]])

st.subheader("Visualizations")

# --- Aggregated/Global Metrics ---
if results_df is None or results_df.empty:
    st.error("No global results found.")
else:
    with st.expander("Aggregated Metrics", expanded=False):
        g_tab1, g_tab2, g_tab3, g_tab4 = st.tabs(["Recall@K", "Weighted nDCG@K", "Mean Latency", "QPS"])
        
        chart_df = results_df.sort_values('dims', ascending=False)
        chart_df['dims_str'] = chart_df['dims'].astype(str)
        
        with g_tab1:
            st.markdown(f"#### Mean Recall@{metric_k} vs Dimensions")
            fig = px.bar(chart_df, y='dims_str', x='recall_mean', orientation='h', color='dims_str',
                         labels={'dims_str': 'Dimensions', 'recall_mean': f'Mean Recall@{metric_k}'}, color_discrete_map=dim_to_color)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Recall@{metric_k}**: Proportion of top-{metric_k} ground truth items found in top-{retrieve_n} results. Higher is better.")
            
        with g_tab2:
            st.markdown(f"#### Mean Weighted nDCG@{metric_k} vs Dimensions")
            fig = px.bar(chart_df, y='dims_str', x='ndcg_mean', orientation='h', color='dims_str',
                         labels={'dims_str': 'Dimensions', 'ndcg_mean': f'Mean nDCG@{metric_k}'}, color_discrete_map=dim_to_color)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Weighted nDCG@{metric_k}**: Ranking quality (weighted top {metric_k} results).")
            st.caption(f"Calculates nDCG@K using the 'true' rank as the relevance score. Items ranked higher by the original embeddings will have higher relevance scores. Essentially, the top item in ground_truth has relevance k, the second k-1, ..., the k-th item has relevance 1.")

        with g_tab3:
            st.markdown("#### Mean Latency (ms) vs Dimensions")
            fig = px.bar(chart_df, y='dims_str', x='mean_latency', orientation='h', color='dims_str',
                         labels={'dims_str': 'Dimensions', 'mean_latency': 'Avg Latency (ms)'}, color_discrete_map=dim_to_color)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Latency**: Average search execution time. Lower is better.")
            
        with g_tab4:
            st.markdown("#### Queries Per Second (QPS)")
            fig = px.bar(chart_df, y='dims_str', x='qps', orientation='h', color='dims_str',
                         labels={'dims_str': 'Dimensions', 'qps': 'Queries Per Second'}, color_discrete_map=dim_to_color)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**QPS**: Throughput. Higher is better.")

# --- Query-by-Query Analysis ---
if details_df is not None and not details_df.empty and len(details_df) > 100:
    with st.expander("Query-by-Query Analysis", expanded=False):
        c1, c2 = st.columns([1, 2])
        use_log = c1.toggle("Log Scale (Y-axis)", value=False)
        p_filter = c2.slider("Latency Percentile Display", 95.0, 100.0, 100.0, step=0.1)

        if p_filter < 100.0:
            q_limit = details_df['latency_ms'].quantile(p_filter / 100.0)
            details_df = details_df[details_df['latency_ms'] <= q_limit].copy()

        q_tab1, q_tab2, q_tab3, q_tab4 = st.tabs(["Latency Timeline", "Latency Distribution", "Warmup Stats", "Recall Distribution"])

        with q_tab1:
            view_type = st.segmented_control("Plot Style", ["Scatter Plot", "Line Plot"], default="Scatter Plot")
            
            if view_type == "Scatter Plot":
                st.markdown("#### Query Latency over Time")
                fig_raw = px.scatter(
                    details_df.sort_values(['label', 'query_index']),
                    x='query_index', y='latency_ms', color='label', facet_row='label',
                    labels={'query_index': 'Query Index', 'latency_ms': 'Latency (ms)', 'label': 'Space'},
                    log_y=use_log, opacity=0.8, height=800, render_mode='webgl', color_discrete_map=color_map
                )
                fig_raw.update_traces(marker=dict(size=3))
                fig_raw.update_yaxes(matches=None)
                fig_raw.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(fig_raw, use_container_width=True)
            else:
                st.markdown("#### Query Latency over Time")
                fig_raw = px.line(
                    details_df.sort_values(['label', 'query_index']),
                    x='query_index', y='latency_ms', color='label',
                    labels={'query_index': 'Query Index', 'latency_ms': 'Latency (ms)', 'label': 'Space'},
                    log_y=use_log, render_mode='webgl', color_discrete_map=color_map
                )
                st.plotly_chart(fig_raw, use_container_width=True)

            st.divider()
            st.markdown("#### Moving Average (Window=50)")
            pivoted_df = pd.DataFrame()
            for label in labels:
                subset = details_df[details_df['label'] == label].sort_values('query_index').set_index('query_index')
                pivoted_df[label] = subset['latency_ms'].rolling(window=50).mean()
            
            fig_rolling = px.line(pivoted_df.dropna(how='all'), 
                                   labels={'index': 'Query Index', 'value': 'Rolling Latency (ms)', 'variable': 'Space'},
                                   log_y=use_log, color_discrete_map=color_map)
            st.plotly_chart(fig_rolling, use_container_width=True)

        with q_tab2:
            st.markdown("#### Latency Distribution by Space")
            viz_df = details_df[details_df['latency_ms'] < details_df['latency_ms'].quantile(0.99)].copy()
            try:
                import altair as alt
                chart = alt.Chart(viz_df).transform_density(
                    'latency_ms', as_=['latency_ms', 'density'], groupby=['label']
                ).mark_area(opacity=0.5).encode(
                    x=alt.X('latency_ms:Q', title='Latency (ms)'),
                    y='density:Q',
                    color=alt.Color('label:N', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())))
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render distribution: {e}")

        with q_tab3:
            if warmup_cutoff > 0:
                st.markdown(f"#### Warmup vs Steady State Statistics (Warmup={warmup_cutoff})")
                st.caption(f"Steady state is calculated after query {warmup_cutoff}.")
            else:
                st.markdown("#### Steady State Statistics")
                st.caption("No warmup was used for this run; all queries are considered steady-state.")

            stats = []
            for label in labels:
                subset = details_df[details_df['label'] == label]
                mean_all = subset['latency_ms'].mean()
                
                # Steady state: after warmup
                steady_subset = subset[subset['query_index'] > warmup_cutoff]
                if not steady_subset.empty:
                    mean_steady, std_steady = steady_subset['latency_ms'].mean(), steady_subset['latency_ms'].std()
                else:
                    mean_steady, std_steady = mean_all, subset['latency_ms'].std()
                
                row = {
                    'Space': label,
                    'Steady State Mean Latency (ms)': round(mean_steady, 3),
                    'Steady State Latency stdev (ms)': round(std_steady, 3)
                }
                
                if warmup_cutoff > 0:
                    row['Mean Latency (ms)'] = round(mean_all, 3)
                    warmup_penalty = ((mean_all - mean_steady) / mean_steady) * 100 if mean_steady > 0 else 0
                    row['Warmup Penalty (%)'] = round(warmup_penalty, 1)
                    
                stats.append(row)
                
            st.dataframe(pd.DataFrame(stats), use_container_width=True)

        with q_tab4:
            st.markdown(f"#### Recall@{metric_k} Distribution by Space")
            try:
                import altair as alt
                # Fill NaNs if any (e.g. for ground truth space itself)
                viz_recall_df = details_df.dropna(subset=['recall']).copy()
                if not viz_recall_df.empty:
                    chart = alt.Chart(viz_recall_df).transform_density(
                        'recall', as_=['recall', 'density'], groupby=['label']
                    ).mark_area(opacity=0.5).encode(
                        x=alt.X('recall:Q', title=f'Recall@{metric_k} (0.0 - 1.0)'),
                        y='density:Q',
                        color=alt.Color('label:N', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())))
                    ).properties(height=400)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Recall data unavailable for this selection.")
            except Exception as e:
                st.warning(f"Could not render recall distribution: {e}")

else:
    st.info("Detailed per-query data unavailable or too small.")


# --- PyGWalker ---
if results_df is not None and not results_df.empty:
    st.subheader("PyGWalker Explorer")
    if details_df is not None and not details_df.empty:
        pyg_data = st.segmented_control("Choose Dataset", ["Global Metrics", "Query-by-Query Metrics"], default="Query-by-Query Metrics")
        
        if pyg_data == "Global Metrics":
            pyg_df = results_df
            renderer = StreamlitRenderer(dataset=pyg_df, default_tab="data")
        else:
            pyg_df = details_df
            spec_path = os.path.join(root_dir, "dashboard", "pyg_specs", "query_analysis.json")
            renderer = StreamlitRenderer(dataset=pyg_df, spec=spec_path, default_tab="chart")
    else:
        pyg_df = results_df
        renderer = StreamlitRenderer(dataset=pyg_df, default_tab="data")
    
    renderer.explorer()
else:
    st.info("PyGWalker explorer is unavailable because no global results were loaded.")
