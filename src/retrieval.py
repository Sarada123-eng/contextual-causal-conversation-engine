"""Retrieval module - wraps vector_index for semantic search operations."""

from vector_index import semantic_search, load_metadata, load_embedding_model

# Re-export functions for backward compatibility
__all__ = ['semantic_search', 'load_metadata', 'load_embedding_model']
