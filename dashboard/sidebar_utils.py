import streamlit as st
import os
import sys

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from search.stats_utils import get_solr_health

def render_sidebar_health():
    """Renders a simplified health indicator in the sidebar."""
    
    # Force a fresh health check
    health = get_solr_health()
    status = health.get('status', 'UNKNOWN')
    ts = health.get('refresh_time', '--:--:--')
    
    # Debug in sidebar (visible in terminal logs)
    print(f"UI SIDEBAR: status={status}, ts={ts}")
    
    # Robust display logic
    if status == 'ONLINE':
        st.sidebar.success(f"🟢 Solr Online")
    elif status == 'DOWN':
        st.sidebar.error(f"🔴 Solr Down (Docker?)")
    else:
        st.sidebar.warning(f"🟡 Solr Unreachable")

    # Use a more explicit string for the timestamp
    st.sidebar.caption(f"Last update: {ts}")
    
    if st.sidebar.button("Refresh Status", key="sidebar_refresh", use_container_width=True, icon=":material/refresh:"):
        st.rerun()
    
    return health
