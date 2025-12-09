"""
Simple Streamlit app to test DuckDB connection.
"""

import streamlit as st
from backend.database import ChroniclesDatabase
from config import DEFAULT_DB_PATH

st.title("Database Connection Test")

db_path = DEFAULT_DB_PATH

db = ChroniclesDatabase(db_path)

if db.test_connection():
    st.success(f"✅ Connected to database: {db_path}")
else:
    st.error(f"❌ Failed to connect to database: {db_path}")