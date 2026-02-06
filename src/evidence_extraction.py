"""Extract and rank evidence turns from retrieved chunks for causal grounding."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class EvidenceTurn(NamedTuple):
	"""Single turn evidence with focused span precision and temporal causality metadata."""
	transcript_id: str
	turn_id: int  # Individual turn index (0-based)
	turn_range: str  # e.g., "3-5" for context
	central_turn_text: str  # The most relevant single turn
	span_text: str  # Full span including ±1 surrounding turns
	similarity_score: float  # Score of central turn
	rank_in_chunk: int  # Rank within this chunk (1-2)
	temporal_label: str = "Unknown"  # "Pre-outcome causal evidence" or "Post-outcome contextual evidence"
	outcome_detected: bool = False  # Whether outcome pattern detected in this span
	outcome_type: str = ""  # Type of outcome (refund, escalation, resolution, etc.)


def detect_outcome_event(turn_text: str) -> tuple[bool, str]:
	"""Detect outcome events in turn text (resolution, refund, escalation, etc.).
	
	Looks for keywords indicating end-state outcomes.
	
	Args:
		turn_text: Single dialogue turn
	
	Returns:
		Tuple of (outcome_detected, outcome_type)
	"""
	text_lower = turn_text.lower()
	
	# Outcome patterns: (keyword_list, outcome_type)
	outcome_patterns = [
		(["refund", "refunded", "reimbursement"], "refund"),
		(["escalat", "escalate", "senior agent"], "escalation"),
		(["resolved", "resolve", "taken care"], "resolution"),
		(["replacement", "replace", "new item"], "replacement"),
		(["credit", "credit issued"], "credit"),
		(["apology", "apologize"], "apology"),
		(["compensation"], "compensation"),
		(["accept", "accepted", "will help"], "acceptance"),
	]
	
	for keywords, outcome_type in outcome_patterns:
		for keyword in keywords:
			if keyword in text_lower:
				return True, outcome_type
	
	return False, ""


def parse_chunk_turns(chunk_text: str) -> list[dict]:
	"""Parse chunk_text back into individual dialogue turns.
	
	Expects format: "Speaker: Text\nSpeaker: Text\n..."
	
	Args:
		chunk_text: Formatted chunk text
	
	Returns:
		List of dicts with keys: speaker, text, turn_text, outcome_detected, outcome_type
	"""
	turns = []
	for line in chunk_text.strip().split("\n"):
		line = line.strip()
		if not line:
			continue
		
		# Parse "Speaker: Text"
		if ":" in line:
			speaker, text = line.split(":", 1)
			speaker = speaker.strip()
			text = text.strip()
			
			# Detect outcome in this turn
			outcome_detected, outcome_type = detect_outcome_event(text)
			
			turns.append({
				"speaker": speaker,
				"text": text,
				"turn_text": line,  # Full formatted turn
				"outcome_detected": outcome_detected,
				"outcome_type": outcome_type,
			})
	
	return turns


def embed_turns(
	model: SentenceTransformer,
	turns: list[dict],
) -> np.ndarray:
	"""Embed all turns in a chunk.
	
	Args:
		model: SentenceTransformer model
		turns: List of turn dicts with 'text' key
	
	Returns:
		NumPy array of embeddings (n_turns, embedding_dim)
	"""
	texts = [turn["text"] for turn in turns]
	embeddings = model.encode(texts, convert_to_numpy=True)
	return embeddings


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
	"""Normalize a single embedding vector.
	
	Args:
		embedding: 1D or 2D embedding array
	
	Returns:
		Normalized embedding
	"""
	if embedding.ndim == 1:
		embedding = embedding.reshape(1, -1)
	
	norm = np.linalg.norm(embedding, axis=1, keepdims=True)
	norm = np.maximum(norm, 1e-8)
	return embedding / norm


def score_turns_against_query(
	query: str,
	turns: list[dict],
	turn_embeddings: np.ndarray,
	model: SentenceTransformer,
) -> list[dict]:
	"""Score each turn by cosine similarity to the query.
	
	Args:
		query: User query string
		turns: List of turn dicts
		turn_embeddings: Turn embeddings array
		model: SentenceTransformer model
	
	Returns:
		List of turns with 'similarity_score' added
	"""
	# Encode and normalize query
	query_embedding = model.encode(query, convert_to_numpy=True)
	query_normalized = normalize_embedding(query_embedding)[0]
	
	# Normalize turn embeddings
	turn_embeddings_normalized = normalize_embedding(turn_embeddings)
	
	# Compute cosine similarity (dot product for normalized vectors)
	similarities = turn_embeddings_normalized @ query_normalized
	
	# Add scores to turns
	scored_turns = []
	for turn, score in zip(turns, similarities):
		turn_copy = turn.copy()
		turn_copy["similarity_score"] = float(score)
		scored_turns.append(turn_copy)
	
	return scored_turns


def classify_temporal_causality(
	central_idx: int,
	all_turns: list[dict],
) -> tuple[str, bool, str]:
	"""Classify evidence as pre-outcome or post-outcome based on turn order.
	
	Scans the turn sequence to detect outcome event and determine whether
	the central turn comes before or after it.
	
	Args:
		central_idx: Index of central evidence turn
		all_turns: All turns in chunk (with outcome_detected flags)
	
	Returns:
		Tuple of (temporal_label, outcome_detected, outcome_type)
	"""
	# Find first outcome in the chunk
	first_outcome_idx = None
	first_outcome_type = ""
	for idx, turn in enumerate(all_turns):
		if turn.get("outcome_detected", False):
			first_outcome_idx = idx
			first_outcome_type = turn.get("outcome_type", "")
			break
	
	# Classify temporal relationship
	if first_outcome_idx is None:
		# No outcome detected in chunk - assume causal
		label = "Pre-outcome causal evidence"
		outcome_detected = False
		outcome_type = ""
	elif central_idx < first_outcome_idx:
		# Evidence comes before outcome - causal
		label = "Pre-outcome causal evidence"
		outcome_detected = False
		outcome_type = ""
	else:
		# Evidence comes after outcome - contextual/confirmatory
		label = "Post-outcome contextual evidence"
		outcome_detected = True
		outcome_type = first_outcome_type
	
	return label, outcome_detected, outcome_type


def extract_evidence(
	query: str,
	chunk_text: str,
	transcript_id: str,
	model: SentenceTransformer,
	top_k: int = 2,
	context_window: int = 1,
) -> list[EvidenceTurn]:
	"""Extract and rank the most causally relevant evidence turns.
	
	Improved precision with temporal causality analysis:
	- Returns only top 1-2 most relevant turns per chunk
	- Analyzes turn order to detect causal progression
	- Labels evidence as pre-outcome causal or post-outcome contextual
	- Detects outcome events (refund, escalation, resolution, etc.)
	
	Args:
		query: User query string
		chunk_text: Chunk text from retrieval
		transcript_id: Transcript ID associated with chunk
		model: SentenceTransformer model
		top_k: Number of evidence turns per chunk (default 2 for precision)
		context_window: Number of turns to include before/after (default=1)
	
	Returns:
		List of top-k EvidenceTurn objects with temporal labels
	"""
	# Parse turns from chunk (preserving order)
	turns = parse_chunk_turns(chunk_text)
	if not turns:
		return []
	
	# Embed turns
	turn_embeddings = embed_turns(model, turns)
	
	# Score turns against query
	scored_turns = score_turns_against_query(
		query, turns, turn_embeddings, model
	)
	
	# Sort by similarity (descending), keeping index
	indexed_turns = [
		(idx, turn) for idx, turn in enumerate(scored_turns)
	]
	indexed_turns.sort(key=lambda x: x[1]["similarity_score"], reverse=True)
	
	# Build EvidenceTurn objects with improved precision
	evidence = []
	for rank_in_chunk, (central_idx, scored_turn) in enumerate(indexed_turns[:top_k], start=1):
		# Define span boundaries (±context_window)
		start_idx = max(0, central_idx - context_window)
		end_idx = min(len(turns), central_idx + context_window + 1)
		
		# Extract span turns in order
		span_turns = turns[start_idx:end_idx]
		span_text = "\n".join([t["turn_text"] for t in span_turns])
		
		# Turn range (1-indexed for readability)
		turn_range = f"{start_idx + 1}-{end_idx}"
		
		# Classify temporal causality
		temporal_label, outcome_detected, outcome_type = classify_temporal_causality(
			central_idx, turns
		)
		
		evidence.append(
			EvidenceTurn(
				transcript_id=transcript_id,
				turn_id=central_idx,  # Precise turn index
				turn_range=turn_range,
				central_turn_text=scored_turn["turn_text"],
				span_text=span_text,
				similarity_score=scored_turn["similarity_score"],
				rank_in_chunk=rank_in_chunk,  # 1 or 2
				temporal_label=temporal_label,
				outcome_detected=outcome_detected,
				outcome_type=outcome_type,
			)
		)
	
	return evidence


def extract_evidence_from_results(
	query: str,
	retrieval_results: pd.DataFrame,
	model: SentenceTransformer,
	top_k: int = 2,
	context_window: int = 1,
) -> list[EvidenceTurn]:
	"""Extract the most causally relevant evidence from multiple chunks.
	
	Improved precision: Returns only top 1-2 evidence turns per chunk
	for focused, high-quality evidence grounding.
	
	Args:
		query: User query string
		retrieval_results: DataFrame from semantic_search with 'chunk_text' column
		model: SentenceTransformer model
		top_k: Number of evidence turns per chunk (default 2 for precision)
		context_window: Number of turns to include before/after (default=1)
	
	Returns:
		List of EvidenceTurn objects, ranked by similarity
	"""
	all_evidence = []
	
	for _, row in retrieval_results.iterrows():
		evidence = extract_evidence(
			query=query,
			chunk_text=row["chunk_text"],
			transcript_id=row["transcript_id"],
			model=model,
			top_k=top_k,
			context_window=context_window,
		)
		all_evidence.extend(evidence)
	
	# Sort all evidence by similarity score (descending)
	all_evidence.sort(key=lambda e: e.similarity_score, reverse=True)
	
	print(f"[Precision] Extracted {len(all_evidence)} high-quality evidence turns")
	print(f"[Precision] From {len(retrieval_results)} chunks × {top_k} turns/chunk max")
	
	# Statistics on temporal causality
	pre_outcome = sum(1 for e in all_evidence if e.temporal_label == "Pre-outcome causal evidence")
	post_outcome = sum(1 for e in all_evidence if e.temporal_label == "Post-outcome contextual evidence")
	if pre_outcome + post_outcome > 0:
		print(f"[Temporal] Pre-outcome causal: {pre_outcome} | Post-outcome contextual: {post_outcome}")
	
	return all_evidence


def print_evidence(evidence: list[EvidenceTurn]) -> None:
	"""Print evidence spans in a readable format with temporal causality labels.
	
	Args:
		evidence: List of EvidenceTurn objects
	"""
	for rank, turn_evidence in enumerate(evidence, start=1):
		temporal_note = f" [{turn_evidence.temporal_label}]"
		if turn_evidence.outcome_detected:
			temporal_note += f" (Outcome: {turn_evidence.outcome_type})"
		
		print(
			f"[{rank}] Score: {turn_evidence.similarity_score:.4f}{temporal_note}\n"
			f"  Transcript: {turn_evidence.transcript_id}\n"
			f"  Turn ID: {turn_evidence.turn_id} | Range: {turn_evidence.turn_range}\n"
			f"  Central Turn: {turn_evidence.central_turn_text}\n"
			f"  Context Span:\n"
		)
		for line in turn_evidence.span_text.split("\n"):
			print(f"    {line}")
		print()


if __name__ == "__main__":
	# Example: Load model and do a demo extraction
	model = SentenceTransformer("all-MiniLM-L6-v2")
	
	# Mock chunk text
	chunk_text = (
		"Agent: Hello, how can I help?\n"
		"Customer: I want to return an item.\n"
		"Agent: What's the reason for return?\n"
		"Customer: It's defective and I need a refund.\n"
		"Agent: I can help with that."
	)
	
	query = "Why do customers request refunds?"
	
	evidence = extract_evidence(
		query=query,
		chunk_text=chunk_text,
		transcript_id="t-001",
		model=model,
		top_k=3,
		context_window=1,
	)
	
	print("=" * 60)
	print("EVIDENCE EXTRACTION")
	print("=" * 60 + "\n")
	print(f"Query: {query}\n")
	print_evidence(evidence)
