import os

# Default Configuration
DEFAULT_DB_PATH = "backend/chronicles_vector.duckdb"
DEFAULT_TOP_K = 10
DEFAULT_MAX_ITER = 3
DEFAULT_MODEL = "gpt-4o-mini"

# Model Options
AVAILABLE_MODELS = ["gpt-4o-mini"]

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384