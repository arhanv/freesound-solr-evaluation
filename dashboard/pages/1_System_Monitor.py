import streamlit as st
from streamlit_extras.row import row
from streamlit_extras.floating_button import floating_button
import time
import pandas as pd
import requests
import plotly.express as px

import os
import sys

# Add project root to sys.path so 'search' package is found
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from search.configs import SOLR_URL
from search.stats_utils import get_content_distribution, get_similarity_spaces
from search.generate_and_index_synthetics import cleanup_synthetic, SYNTHETIC_CLEANUP_QUERY
from search.pca import delete_pca_vectors
from search.index_to_solr import SolrIndexer
from sidebar_utils import render_sidebar_health

st.set_page_config(
    page_title="Freesound Solr Monitor",
    layout="centered",
)

health = render_sidebar_health()

st.title("Freesound Solr Monitor")

st.subheader("Status")
st.caption(f"Collection: {health.get('collection', 'Unknown')} | Last Refresh: {health.get('refresh_time', '--:--:--')}")

c1, c2, c3 = st.columns(3)

status = health.get("status", "UNKNOWN")
if status == "ONLINE":
    d_val = "● Solr is running"
    d_color = "normal"   # Green
elif status == "UNKNOWN":
    d_color = "off"      # Gray/Yellow-ish (Streamlit's 'off' is neutral)
    d_val = "○ Unable to reach Solr"
else: # DOWN
    d_val = "● Check if Docker is running"
    d_color = "inverse"  # Red

current_docs = health.get('num_docs', 0)
current_size = health.get('size_mb', 0)

if 'prev_num_docs' not in st.session_state:
    st.session_state['prev_num_docs'] = current_docs
if 'prev_size_mb' not in st.session_state:
    st.session_state['prev_size_mb'] = current_size

delta_docs = current_docs - st.session_state['prev_num_docs']
delta_size = current_size - st.session_state['prev_size_mb']

# Update session state for the next run
st.session_state['prev_num_docs'] = current_docs
st.session_state['prev_size_mb'] = current_size

c1.metric(label="Status", value=status, delta=d_val, delta_color=d_color)
c2.metric("Docs", f"{current_docs:,}", delta=delta_docs if delta_docs != 0 else None)
c3.metric("Total Index Size", f"{current_size} MB", delta=f"{delta_size:.2f} MB" if delta_size != 0 else None)

if status != "ONLINE":
    st.error(f"**Connection Error:** {health.get('error', 'Unknown issue')}")
st.divider()


# Similarity Spaces & Visualization
st.subheader("Similarity Spaces")
spaces = get_similarity_spaces()

if spaces:
    df_spaces = pd.DataFrame(spaces)

    # Create grouping columns for better visualization
    def get_source(name):
        return name.split('_pca')[0].split('_synthetic')[0]
    
    df_spaces['source_space'] = df_spaces['name'].apply(get_source)
    
    # Generate Multi-tags
    def generate_tags(row):
        t = []
        t.append("synthetic-data" if "synthetic" in row['name'] else "real-data")
        t.append("projected-space" if "_pca" in row['name'] else "source-space")
        return t
    
    df_spaces['tags'] = df_spaces.apply(generate_tags, axis=1)
    df_spaces['is_source'] = df_spaces['name'].apply(lambda x: "_pca" not in x)
    df_spaces = df_spaces.sort_values(by=['source_space', 'is_source', 'dimension'], ascending=[True, False, False]).reset_index(drop=True)

    # 1. Detailed Table
    cols = ["name", "dimension", "count", "size_mb", "tags"]
    df_display = df_spaces[cols]

    def highlight_source(row):
        is_source = "source-space" in row["tags"]
        color = "rgba(241, 196, 15, 0.15)" if is_source else ""
        return [f"background-color: {color}" if color else ""] * len(row)

    styled_df = df_display.style.apply(highlight_source, axis=1)

    event = st.dataframe(
        styled_df,
        column_config={
            "name": "Space Name",
            "dimension": "Dimensions",
            "count": "Vector Count",
            "size_mb": "Est. Size (MB)",
            "tags": st.column_config.MultiselectColumn(
                "Classification",
                options=["real-data", "synthetic-data", "source-space", "projected-space"],
                color="auto"
            )
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_space = df_display.iloc[selected_idx]
        space_name = selected_space["name"]
        num_vecs = selected_space["count"]
        tags = selected_space["tags"]
        
        # Omit options if it is real-data source-space
        if "source-space" in tags and "real-data" in tags:
            pass # No actions shown for real data source spaces
        else:
            st.markdown(f"**Selected Space:** `{space_name}`")
            act_col1, act_col2 = st.columns([3, 1]) # Right align the actions
            
            with act_col2:
                with st.popover(f"Delete Space", use_container_width=True, icon=":material/delete:"):
                    st.markdown(f"**Delete `{space_name}`**\n\nThis will remove **{num_vecs:,}** vector documents from Solr.")
                    
                    # 1. Show the checkbox BEFORE the button so the user can decide
                    model_path = os.path.join(root_dir, "models", "pca", f"{space_name}.pkl")
                    delete_model_checked = False
                    
                    if "_pca" in space_name and os.path.exists(model_path):
                        delete_model_checked = st.checkbox("Also delete saved .pkl model?", key=f"del_model_{space_name}")

                    # 2. The Delete Button
                    if st.button("Yes, delete everything selected", type="primary", key=f"del_vecs_{space_name}"):
                        with st.spinner("Processing deletion..."):
                            try:
                                # --- PART A: Delete Vectors (Always happens) ---
                                if "_pca" in space_name:
                                    import builtins
                                    original_input = builtins.input
                                    builtins.input = lambda _: 'y'
                                    try:
                                        delete_pca_vectors(space_name, solr_url=SOLR_URL)
                                    finally:
                                        builtins.input = original_input
                                elif "synthetic" in space_name:
                                    cleanup_synthetic(SOLR_URL)
                                
                                # --- PART B: Delete Files (Only if checkbox was checked) ---
                                if delete_model_checked:
                                    os.remove(model_path)
                                    json_path = model_path.replace(".pkl", ".json")
                                    if os.path.exists(json_path):
                                        os.remove(json_path)
                                    st.toast("Model files removed!")

                                st.toast(f"Deleted vectors for {space_name}!")
                                time.sleep(1)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Failed: {e}")

    # 2. Vector Count Distribution (Horizontal Grouped Bar)
    st.subheader("Distribution of Vectors by Similarity Space")
    fig_bar = px.bar(
        df_spaces,
        y="name",
        x="count",
        color="dimension",  # Gradient within group
        facet_row="source_space", # Grouping by source
        labels={"name": "Space", "count": "Vector Count", "dimension": "Dimensions"},
        orientation='h',
        color_continuous_scale="Viridis",
        category_orders={"source_space": sorted(df_spaces['source_space'].unique())}
    )
    fig_bar.update_layout(height=400 + (len(df_spaces['source_space'].unique()) * 50), showlegend=False)
    fig_bar.update_yaxes(matches=None) # Allow unique labels per facet
    st.plotly_chart(fig_bar, width='stretch')

    # 3. Index Size Chart (Bubble chart: Dim vs Size, bubble size = Count)
    st.subheader("Storage Efficiency: Dimensions vs Estimated Index Size")
    fig_scatter = px.scatter(
        df_spaces,
        x="dimension",
        y="size_mb",
        size="count",
        color="source_space", # Color groups by source
        hover_name="name",
        labels={"dimension": "Embedding Dims", "size_mb": "Index Size (MB)", "source_space": "Source"},
        size_max=20,
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
    )
    st.plotly_chart(fig_scatter, width='stretch')
    st.caption("This uses an estimation of the index size based on the number of vectors and their dimensions.")
else:
    st.info("No similarity spaces found.")

st.markdown("### Actions")

c1, c2, c3 = st.columns(3)

with c1:
    with st.popover("Clean Up Synthetics", width='stretch', icon=":material/layers_clear:"):
        st.markdown("**Purge all synthetic vectors & parents.**\n\nRemoves everything generated via GMM, including PCA-reduced versions of synthetic data.")
        
        # Calculate impacts
        try:
            indexer = SolrIndexer(SOLR_URL)
            to_delete = indexer.solr.search(SYNTHETIC_CLEANUP_QUERY, rows=0).hits
            parents_to_delete = indexer.solr.search(f"({SYNTHETIC_CLEANUP_QUERY}) AND content_type:s", rows=0).hits
            vectors_to_delete = indexer.solr.search(f"({SYNTHETIC_CLEANUP_QUERY}) AND content_type:v", rows=0).hits
        except Exception:
            to_delete = parents_to_delete = vectors_to_delete = 0

        if to_delete > 0:
            st.warning(f"This will delete **{to_delete:,}** documents:")
            st.write(f"- {parents_to_delete:,} parent sounds")
            st.write(f"- {vectors_to_delete:,} similarity vectors")
        else:
            st.info("No synthetic data found in index.")

        if st.button("Yes, delete synthetic", type="primary", key="cleanup_synth", disabled=(to_delete == 0)):
            with st.spinner("Deleting synthetic data..."):
                try:
                    cleanup_synthetic(SOLR_URL)
                    st.toast("Synthetic data cleaned up!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")

with c2:
    with st.popover("Optimize Index", width='stretch', icon=":material/compress:"):
        st.markdown("**Force Merge Segments.**\n\nRun this after large deletions to reclaim disk space and speed up search.")
        st.warning("This is a heavy operation. Continue?")
        if st.button("Yes, optimize now", key="optimize_now"):
            with st.spinner("Triggering optimization..."):
                try:
                    requests.get(f"{SOLR_URL}/update?optimize=true", timeout=10)
                    st.toast("Optimization triggered!")
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

with c3:
    with st.popover("Clear Entire Index", width='stretch', icon=":material/delete_forever:", type="primary"):
        st.markdown("**DANGER: Wipe the entire collection.**\n\nThis will remove all real metadata, all synthetic data, and all similarity spaces.")
        st.error("This action is irreversible.")
        if st.button("Yes, CLEAR EVERYTHING", type="primary", key="clear_all"):
            with st.spinner("Clearing all data..."):
                try:
                    indexer = SolrIndexer(SOLR_URL)
                    indexer.clear_index()
                    st.toast("Index cleared successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Clear failed: {e}")

auto_refresh = st.segmented_control(label="Auto Refresh? (5s)",
    options=["Yes", "No"],
    default="No", width="stretch")
if auto_refresh == "Yes":
    time.sleep(5)
    st.rerun()
