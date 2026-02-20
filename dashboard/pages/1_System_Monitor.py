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
from search.stats_utils import get_solr_health, get_content_distribution, get_similarity_spaces
from search.generate_and_index_synthetics import cleanup_synthetic, SYNTHETIC_CLEANUP_QUERY
from search.index_to_solr import SolrIndexer

st.set_page_config(
    page_title="Freesound Solr Monitor",
    layout="centered",
)

floating_button(label="Refresh", icon=":material/refresh:", on_click=lambda: st.rerun())

st.title("Freesound Solr Monitor")

st.subheader("Status")
health = get_solr_health()
st.caption(f"Collection: {health.get('collection', 'Unknown')}")

c1, c2, c3 = st.columns(3)
c1.metric("Status", health.get("status", "Unknown"))
c2.metric("Docs", f"{health.get('num_docs', 0):,}")
c3.metric("Size", f"{health.get('size_mb', 0)} MB")
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
    df_spaces = df_spaces.sort_values(['source_space', 'dimension'])

    # 1. Detailed Table
    cols = ["name", "dimension", "count", "size_mb", "tags"]
    st.dataframe(
        df_spaces[cols],
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
        hide_index=True,
        use_container_width=True
    )

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
    st.plotly_chart(fig_bar, use_container_width=True)

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
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("This uses an estimation of the index size based on the number of vectors and their dimensions.")
else:
    st.info("No similarity spaces found.")

st.markdown("### Actions")

c1, c2, c3 = st.columns(3)

with c1:
    with st.popover("Clean Up Synthetics", use_container_width=True, icon=":material/layers_clear:"):
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
    with st.popover("Optimize Index", use_container_width=True, icon=":material/compress:"):
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
    with st.popover("Clear Entire Index", use_container_width=True, icon=":material/delete_forever:", type="primary"):
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
if auto_refresh:
    time.sleep(5)
    st.rerun()
