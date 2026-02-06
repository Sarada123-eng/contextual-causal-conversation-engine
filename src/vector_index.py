"""Build and manage FAISS vector index for chunk embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def load_embeddings(embeddings_path: str | Path) -> np.ndarray:
	"""Load embeddings from .npy file.
	
	Args:
		embeddings_path: Path to chunk_embeddings.npy
	
	Returns:
		NumPy array of shape (n_chunks, embedding_dim)
	"""
	embeddings_path = Path(embeddings_path)
	if not embeddings_path.exists():
		raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
	
	embeddings = np.load(embeddings_path)
	print(f"Loaded embeddings: {embeddings.shape}")
	return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
	"""Normalize embeddings using L2 norm for cosine similarity.
	
	Args:
		embeddings: NumPy array of shape (n_chunks, embedding_dim)
	
	Returns:
		Normalized embeddings (unit vectors)
	"""
	norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
	norms = np.maximum(norms, 1e-8)  # Avoid division by zero
	normalized = embeddings / norms
	print(f"✓ Normalized embeddings for cosine similarity")
	return normalized


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
	"""Build FAISS index using IndexFlatIP (inner product for normalized vectors).
	
	For normalized embeddings, IndexFlatIP computes cosine similarity.
	
	Args:
		embeddings: Normalized embeddings of shape (n_chunks, embedding_dim)
	
	Returns:
		FAISS IndexFlatIP index
	"""
	embedding_dim = embeddings.shape[1]
	index = faiss.IndexFlatIP(embedding_dim)
	index.add(embeddings)
	print(f"✓ FAISS index created (IndexFlatIP, {embedding_dim}D)")
	print(f"✓ Indexed vectors: {index.ntotal}")
	return index


def save_faiss_index(
	index: faiss.IndexFlatIP,
	output_path: str | Path,
) -> None:
	"""Save FAISS index to disk.
	
	Args:
		index: FAISS index object
		output_path: Path to save index (e.g., data/faiss_index.bin)
	"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(output_path))
	print(f"✓ Index saved: {output_path}")


def load_faiss_index(index_path: str | Path) -> faiss.IndexFlatIP:
	"""Load FAISS index from disk.
	
	Args:
		index_path: Path to faiss_index.bin
	
	Returns:
		FAISS index object
	"""
	index_path = Path(index_path)
	if not index_path.exists():
		raise FileNotFoundError(f"Index file not found: {index_path}")
	
	index = faiss.read_index(str(index_path))
	print(f"✓ Index loaded: {index_path} ({index.ntotal} vectors)")
	return index


def search(
	index: faiss.IndexFlatIP,
	query_embedding: np.ndarray,
	k: int = 8,  # Increased from 5 to 8 for broader retrieval breadth
) -> tuple[np.ndarray, np.ndarray]:
	"""Search for similar chunks using FAISS index.
	
	Args:
		index: FAISS index
		query_embedding: Single embedding or batch of embeddings (normalized)
		k: Number of nearest neighbors to return (default 8 for broader evidence)
	
	Returns:
		Tuple of (distances, indices) where:
		- distances: Similarity scores (cosine similarity for normalized vectors)
		- indices: Chunk indices of the k nearest neighbors
	"""
	if query_embedding.ndim == 1:
		query_embedding = query_embedding.reshape(1, -1)
	
	distances, indices = index.search(query_embedding, k)
	return distances, indices


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
	"""Load SentenceTransformer model for encoding queries.
	
	Args:
		model_name: HuggingFace model identifier
	
	Returns:
		Loaded SentenceTransformer model
	"""
	model = SentenceTransformer(model_name)
	return model


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
	"""Encode a query string to embedding and normalize.
	
	Args:
		model: SentenceTransformer model
		query: User query string
	
	Returns:
		Normalized embedding
	"""
	embedding = model.encode(query, convert_to_numpy=True)
	norm = np.linalg.norm(embedding)
	norm = max(norm, 1e-8)
	normalized = embedding / norm
	return normalized.reshape(1, -1)


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
	"""Load chunk metadata CSV.
	
	Args:
		metadata_path: Path to chunk_metadata.csv
	
	Returns:
		DataFrame with metadata
	"""
	metadata_path = Path(metadata_path)
	if not metadata_path.exists():
		raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
	return pd.read_csv(metadata_path)


def semantic_search(
	query: str,
	index_path: str | Path = Path("data") / "faiss_index.bin",
	metadata_path: str | Path = Path("data") / "chunk_metadata.csv",
	k: int = 8,  # Default top_k for broader retrieval breadth
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> pd.DataFrame:
	"""Perform semantic search on conversation chunks with enhanced breadth.
	
	Retrieves top-k chunks for comprehensive evidence collection. Supports
	dynamic k parameter override for flexible search breadth control.
	
	Args:
		query: User query string
		index_path: Path to FAISS index (default: data/faiss_index.bin)
		metadata_path: Path to chunk metadata CSV (default: data/chunk_metadata.csv)
		k: Number of results to return (default 8 for broader evidence diversity)
		   Override per-call: semantic_search(query, k=5) or k=15
		model_name: Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
	
	Returns:
		DataFrame with top-k results including:
		- All metadata columns (chunk_id, transcript_id, start_turn, end_turn, etc.)
		- rank: Result position (1-k)
		- similarity_score: Cosine similarity (0-1 scale)
	
	Raises:
		FileNotFoundError: If index or metadata files not found
	"""
	print(f"Query: {query}")
	print(f"Searching for {k} most similar chunks...\n")
	
	# Load model and encode query
	model = load_embedding_model(model_name)
	query_embedding = encode_query(model, query)
	
	# Load index
	index = load_faiss_index(index_path)
	
	# Search
	distances, indices = search(index, query_embedding, k=k)
	
	# Load metadata
	metadata = load_metadata(metadata_path)
	
	# Build results DataFrame
	results = []
	for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
		row = metadata.iloc[idx].to_dict()
		row["rank"] = rank
		row["similarity_score"] = float(distance)
		results.append(row)
	
	results_df = pd.DataFrame(results)
	
	# Enhanced retrieval diagnostics printing
	num_chunks = len(results_df)
	num_transcripts = results_df["transcript_id"].nunique()
	avg_similarity = results_df["similarity_score"].mean()
	min_similarity = results_df["similarity_score"].min()
	max_similarity = results_df["similarity_score"].max()
	
	print(f"[Breadth] Retrieved {num_chunks} chunks (k={k}) from {num_transcripts} unique transcripts")
	print(f"[Diversity] Similarity range: {min_similarity:.4f} - {max_similarity:.4f}")
	print(f"[Quality] Average similarity: {avg_similarity:.4f}")
	
	return results_df


def print_search_results(results_df: pd.DataFrame, show_summary: bool = True) -> None:
	"""Print search results in a readable format.
	
	Args:
		results_df: DataFrame returned by semantic_search()
		show_summary: Whether to print summary statistics
	"""
	if show_summary:
		print("\n" + "=" * 70)
		print("SEMANTIC SEARCH RESULTS SUMMARY")
		print("=" * 70)
		print(f"Total chunks retrieved: {len(results_df)}")
		print(f"Unique transcripts: {results_df['transcript_id'].nunique()}")
		print(f"Similarity range: {results_df['similarity_score'].min():.4f} - {results_df['similarity_score'].max():.4f}")
		print(f"Average similarity: {results_df['similarity_score'].mean():.4f}")
		print("\n" + "=" * 70 + "\n")
	
	for _, row in results_df.iterrows():
		print(
			f"[{row['rank']}] Similarity: {row['similarity_score']:.4f}\n"
			f"  chunk_id: {row['chunk_id']}\n"
			f"  transcript_id: {row['transcript_id']}\n"
			f"  turns: {row['start_turn']}-{row['end_turn']}\n"
		)



def main() -> None:
	"""Main pipeline: load embeddings, build and save FAISS index."""
	# Paths
	embeddings_path = Path("data") / "chunk_embeddings.npy"
	index_output = Path("data") / "faiss_index.bin"
	
	# Load embeddings
	embeddings = load_embeddings(embeddings_path)
	
	# Normalize for cosine similarity
	normalized_embeddings = normalize_embeddings(embeddings)
	
	# Build FAISS index
	index = build_faiss_index(normalized_embeddings)
	
	# Save index
	save_faiss_index(index, index_output)
	
	print(f"\n✓ FAISS vector index pipeline complete!")


if __name__ == "__main__":
	# Uncomment to build/rebuild the index
	# main()
	
	# Semantic search example with improved breadth
	query = "Why do customers request refunds?"
	results = semantic_search(query, k=8)  # Retrieve 8 chunks for better diversity
	print("\n" + "="*70)
	print("SEARCH RESULTS (Top 8 - Improved Diversity)")
	print("="*70)
	print_search_results(results, show_summary=True)
