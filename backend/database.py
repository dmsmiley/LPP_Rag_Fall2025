import duckdb
import os
from sentence_transformers import SentenceTransformer
import streamlit as st
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class ChroniclesDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model = self._load_model()

    @staticmethod
    @st.cache_resource
    def _load_model():
        return SentenceTransformer(EMBEDDING_MODEL_NAME)

    def test_connection(self) -> bool:
        """Test if the database can be connected to."""
        if not os.path.exists(self.db_path):
            return False
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            conn.close()
            return True
        except Exception:
            return False

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Query the database for relevant passages.
        
        Args:
            query_text: The search query.
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing 'text' and 'similarity'.
        """

        try:
            # Connect to database in read-only mode
            # read_only=True prevents accidental modifications
            conn = duckdb.connect(self.db_path, read_only=True)
            
            # Convert query text to embedding vector
            query_embedding = self.model.encode(query_text).tolist()
            
            # Execute vector search
            # Return top k most similar passages
            # Note: We cast the parameter to FLOAT[384] to match the embedding dimension
            results = conn.execute(f"""
                SELECT text, array_cosine_similarity(embedding, ?::FLOAT[{EMBEDDING_DIMENSION}]) as similarity
                FROM chr_rag_documents
                ORDER BY similarity DESC
                LIMIT ?
            """, [query_embedding, top_k]).fetchall()
            
            # Close database connection
            conn.close()
            
            # Format results for the agent
            return [{"text": row[0], "similarity": float(row[1])} for row in results]
            
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")
