
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
    layout="centered",
)

render_sidebar_health()

st.title("Freesound Solr Evaluation Dashboard")

st.info("""
This dashboard provides tools to monitor and evaluate the similarity search performance of Freesound's Solr backend.
""")

st.info("Select a page from the sidebar to get started.")
