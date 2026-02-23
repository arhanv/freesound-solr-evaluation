
import os
import sys

# Add project root to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from sidebar_utils import render_sidebar_health

st.set_page_config(
    page_title="Freesound Solr Dashboard",
    page_icon=":material/home:",
    layout="wide",
)

def home_page():
    render_sidebar_health()

    st.title("Freesound Solr Evaluation Dashboard")

    st.markdown("""
    This dashboard provides tools to monitor and evaluate the similarity search performance of Freesound's Solr backend. 
    Use the links below or the sidebar to navigate through the available utilities.
    """)

    st.subheader("Pages")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.page_link("pages/1_System_Monitor.py", label="**System Monitor**", icon=":material/chart_data:")
        st.caption("View Solr health, memory usage, and manage existing similarity spaces.")
    with c2.container(border=True):
        st.page_link("pages/2_Run_Evaluation.py", label="**Run Evaluation Queries**", icon=":material/experiment:")
        st.caption("Execute batched query evaluations to test search latency and retrieval quality.")
    
    c3, c4 = st.columns(2)
    with c3.container(border=True):
        st.page_link("pages/3_Run_PCA.py", label="**Apply PCA**", icon=":material/compress:")
        st.caption("Generate and manage lower-dimensional vector models via PCA.")
    with c4.container(border=True):
        st.page_link("pages/4_Synthetic_Data.py", label="**Synthetic Data**", icon=":material/science:")
        st.caption("Fit a GMM, generate synthetic vectors to stress-test Solr at scale, and visualize distribution quality.")

    c5, _ = st.columns(2)
    with c5.container(border=True):
        st.page_link("pages/5_Analysis.py", label="**Analyze Results**", icon=":material/analytics:")
        st.caption("Visualize and compare evaluation results.")


pg = st.navigation([
    st.Page(home_page, title="Home", icon=":material/home:", default=True),
    st.Page("pages/1_System_Monitor.py", title="System Monitor", icon=":material/chart_data:"),
    st.Page("pages/2_Run_Evaluation.py", title="Run Evaluation Queries", icon=":material/experiment:"),
    st.Page("pages/3_Run_PCA.py", title="Apply PCA", icon=":material/compress:"),
    st.Page("pages/4_Synthetic_Data.py", title="Synthetic Data", icon=":material/science:"),
    st.Page("pages/5_Analysis.py", title="Analyze Results", icon=":material/analytics:")
])

pg.run()

