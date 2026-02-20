
import os
import sys

# Add project root to sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st

st.set_page_config(
    page_title="Freesound Solr Dashboard",
    layout="centered",
)

st.title("Freesound Solr Evaluation Dashboard")

st.markdown("""
### Welcome to the Solr Evaluation Suite

This dashboard provides tools to monitor and evaluate the similarity search performance of Freesound's Solr backend.
""")

st.info("Select a page from the sidebar to get started.")
