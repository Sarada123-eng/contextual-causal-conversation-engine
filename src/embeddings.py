"""Generate embeddings for conversation chunks using SentenceTransformers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_chunks(csv_path: str | Path) -> pd.DataFrame:
	"""Load the conversation chunks CSV."""
	csv_path = Path(csv_path)
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")
	return pd.read_csv(csv_path)


def generate_embeddings(
	chunks_df: pd.DataFrame,
	model_name: str = "all-MiniLM-L6-v2",
	text_column: str = "chunk_text",
	show_progress: bool = True,
) -> np.ndarray:
	"""Generate embeddings for the chunk text column.
	
	Args:
		chunks_df: DataFrame containing chunks
		model_name: SentenceTransformer model name
		text_column: Name of the column containing text to embed
		show_progress: Whether to show progress bar during encoding
	
	Returns:
		NumPy array of embeddings with shape (n_chunks, embedding_dim)
	"""
	if text_column not in chunks_df.columns:
		raise ValueError(f"Missing column: {text_column}")
	
	print(f"Loading model: {model_name}")
	model = SentenceTransformer(model_name)
	
	# Prepare texts (fill NaN, convert to string)
	texts = chunks_df[text_column].fillna("").astype(str).tolist()
	
	print(f"Generating embeddings for {len(texts)} chunks...")
	embeddings = model.encode(
		texts,
		convert_to_numpy=True,
		show_progress_bar=show_progress,
	)
	
	return embeddings


def save_embeddings(
	embeddings: np.ndarray,
	output_path: str | Path,
) -> None:
	"""Save embeddings array to disk as .npy file."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(output_path, embeddings)
	print(f"✓ Embeddings saved: {output_path}")


def save_metadata(
	chunks_df: pd.DataFrame,
	output_path: str | Path,
) -> None:
	"""Save chunk metadata (preserving row order to match embeddings)."""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	chunks_df.to_csv(output_path, index=False)
	print(f"✓ Metadata saved: {output_path}")


def extract_and_save_metadata(
	chunks_csv: str | Path = Path("data") / "conversation_chunks.csv",
	metadata_output: str | Path = Path("data") / "chunk_metadata.csv",
	metadata_columns: list[str] | None = None,
) -> None:
	"""Load chunks CSV and save metadata columns without regenerating embeddings.
	
	Args:
		chunks_csv: Path to conversation_chunks.csv
		metadata_output: Path to save metadata CSV
		metadata_columns: List of columns to extract; if None, uses defaults
	"""
	if metadata_columns is None:
		metadata_columns = ["chunk_id", "transcript_id", "start_turn", "end_turn"]
	
	metadata_output = Path(metadata_output)
	metadata_output.parent.mkdir(parents=True, exist_ok=True)
	
	print(f"Loading chunks from: {chunks_csv}")
	chunks_df = load_chunks(chunks_csv)
	
	# Extract only requested metadata columns
	metadata_df = chunks_df[metadata_columns]
	metadata_df.to_csv(metadata_output, index=False)
	
	print(f"✓ Metadata saved: {metadata_output} ({len(metadata_df)} rows)")


def main() -> None:
	"""Main pipeline: load chunks, generate embeddings, save."""
	# Paths
	chunks_csv = Path("data") / "conversation_chunks.csv"
	embeddings_output = Path("data") / "chunk_embeddings.npy"
	metadata_output = Path("data") / "chunk_metadata.csv"
	
	# Load chunks
	print(f"Loading chunks from: {chunks_csv}")
	chunks_df = load_chunks(chunks_csv)
	print(f"Loaded {len(chunks_df)} chunks")
	
	# Generate embeddings
	embeddings = generate_embeddings(
		chunks_df,
		model_name="all-MiniLM-L6-v2",
		text_column="chunk_text",
		show_progress=True,
	)
	
	# Print shape for verification
	print(f"\nEmbeddings shape: {embeddings.shape}")
	print(f"  - Number of chunks: {embeddings.shape[0]}")
	print(f"  - Embedding dimension: {embeddings.shape[1]}")
	
	# Save outputs
	save_embeddings(embeddings, embeddings_output)
	save_metadata(chunks_df, metadata_output)


if __name__ == "__main__":
	import sys
	
	if len(sys.argv) > 1 and sys.argv[1] == "--metadata-only":
		# Save metadata only (embeddings already exist)
		extract_and_save_metadata(
			chunks_csv=Path("data") / "conversation_chunks.csv",
			metadata_output=Path("data") / "chunk_metadata.csv",
		)
	else:
		# Full pipeline: generate embeddings and save metadata
		main()
