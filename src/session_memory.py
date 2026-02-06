"""Manage analytical session memory for query tracking and evidence accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from factor_normalizer import normalize_factor


@dataclass
class QueryRecord:
	"""Single query with associated results."""
	query: str
	timestamp: str
	retrieval_results: Optional[pd.DataFrame] = None
	num_chunks_retrieved: int = 0
	num_evidence_turns: int = 0
	factors_identified: List[str] = field(default_factory=list)


@dataclass
class EvidenceRecord:
	"""Single piece of evidence from a transcript."""
	transcript_id: str
	turn_range: str
	span_text: str
	similarity_score: float
	source_query: str


class SessionMemory:
	"""Manages analytical session state and context."""
	
	def __init__(self):
		"""Initialize empty session memory."""
		self.queries: List[QueryRecord] = []
		self.evidence: List[EvidenceRecord] = []
		self.transcript_ids_seen: set[str] = set()
		self.factors_identified: set[str] = set()  # Unique factors across all queries
		self.session_start: str = datetime.now().isoformat()
		self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	
	def add_query(
		self,
		query: str,
		retrieval_results: Optional[pd.DataFrame] = None,
		num_evidence_turns: int = 0,
		factors_identified: Optional[List[str]] = None,
	) -> None:
		"""Add a new query to session history.
		
		Args:
			query: User query string
			retrieval_results: DataFrame from semantic_search (optional)
			num_evidence_turns: Count of evidence turns extracted
			factors_identified: List of causal factor names
		"""
		num_chunks = len(retrieval_results) if retrieval_results is not None else 0
		
		# Update global factors set (only if new evidence found)
		# Normalize factors to canonical categories
		if factors_identified and num_evidence_turns > 0:
			for factor in factors_identified:
				if factor:  # Skip empty strings
					normalized_factor = normalize_factor(factor)
					self.factors_identified.add(normalized_factor)
		
		record = QueryRecord(
			query=query,
			timestamp=datetime.now().isoformat(),
			retrieval_results=retrieval_results,
			num_chunks_retrieved=num_chunks,
			num_evidence_turns=num_evidence_turns,
			factors_identified=factors_identified or [],
		)
		self.queries.append(record)
	
	def add_evidence(
		self,
		transcript_id: str,
		turn_range: str,
		span_text: str,
		similarity_score: float,
		source_query: str,
	) -> None:
		"""Add evidence span to session memory.
		
		Args:
			transcript_id: ID of source transcript
			turn_range: Turn range (e.g., "3-5")
			span_text: Full context span text
			similarity_score: Relevance score
			source_query: Query that retrieved this evidence
		"""
		evidence = EvidenceRecord(
			transcript_id=transcript_id,
			turn_range=turn_range,
			span_text=span_text,
			similarity_score=similarity_score,
			source_query=source_query,
		)
		self.evidence.append(evidence)
		self.transcript_ids_seen.add(transcript_id)
	
	def add_evidence_batch(
		self,
		evidence_turns: List[Any],
		source_query: str,
	) -> None:
		"""Add multiple evidence turns from a list.
		
		Args:
			evidence_turns: List of EvidenceTurn objects
			source_query: Query that retrieved this evidence
		"""
		for turn in evidence_turns:
			self.add_evidence(
				transcript_id=turn.transcript_id,
				turn_range=turn.turn_range,
				span_text=turn.span_text,
				similarity_score=turn.similarity_score,
				source_query=source_query,
			)
	
	def get_context(self) -> Dict[str, Any]:
		"""Get current session context summary.
		
		Returns:
			Dictionary with session statistics and state
		"""
		recent_queries = [q.query for q in self.queries[-5:]]
		
		context = {
			"session_id": self.session_id,
			"session_start": self.session_start,
			"total_queries": len(self.queries),
			"total_evidence_spans": len(self.evidence),
			"unique_transcripts": len(self.transcript_ids_seen),
			"recent_queries": recent_queries,
			"all_factors_identified": sorted(list(self.factors_identified)),  # Use set, sorted for consistent order
			"transcript_ids": sorted(self.transcript_ids_seen),
		}
		return context
	
	def get_query_history(self) -> List[Dict[str, Any]]:
		"""Get full query history as list of dicts.
		
		Returns:
			List of query records
		"""
		history = []
		for q in self.queries:
			history.append({
				"query": q.query,
				"timestamp": q.timestamp,
				"chunks_retrieved": q.num_chunks_retrieved,
				"evidence_turns": q.num_evidence_turns,
				"factors": q.factors_identified,
			})
		return history
	
	def get_evidence_by_transcript(self, transcript_id: str) -> List[EvidenceRecord]:
		"""Get all evidence from a specific transcript.
		
		Args:
			transcript_id: Transcript ID to filter by
		
		Returns:
			List of evidence records from that transcript
		"""
		return [e for e in self.evidence if e.transcript_id == transcript_id]
	
	def get_evidence_by_query(self, query: str) -> List[EvidenceRecord]:
		"""Get all evidence retrieved by a specific query.
		
		Args:
			query: Query string to filter by
		
		Returns:
			List of evidence records from that query
		"""
		return [e for e in self.evidence if e.source_query == query]
	
	def get_all_factors(self) -> List[str]:
		"""Get all unique factors identified across the session.
		
		Returns:
			Sorted list of unique factor names
		"""
		return sorted(list(self.factors_identified))
	
	def has_factor(self, factor: str) -> bool:
		"""Check if a factor has been identified in the session.
		
		Args:
			factor: Factor name to check
		
		Returns:
			True if factor exists in session
		"""
		return factor in self.factors_identified
	
	def get_factor_count(self) -> int:
		"""Get the number of unique factors identified.
		
		Returns:
			Count of unique factors
		"""
		return len(self.factors_identified)
	
	def reset_session(self) -> None:
		"""Clear all session memory and restart."""
		self.queries = []
		self.evidence = []
		self.transcript_ids_seen = set()
		self.factors_identified = set()
		self.session_start = datetime.now().isoformat()
		self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	
	def print_summary(self) -> None:
		"""Print a formatted session summary."""
		context = self.get_context()
		
		print("\n" + "=" * 70)
		print("SESSION MEMORY SUMMARY")
		print("=" * 70)
		print(f"\nSession ID: {context['session_id']}")
		print(f"Started: {context['session_start']}")
		print(f"\nQueries: {context['total_queries']}")
		print(f"Evidence Spans: {context['total_evidence_spans']}")
		print(f"Unique Transcripts: {context['unique_transcripts']}")
		
		if context['recent_queries']:
			print(f"\nRecent Queries:")
			for idx, q in enumerate(context['recent_queries'], start=1):
				print(f"  {idx}. {q}")
		
		if context['all_factors_identified']:
			print(f"\nFactors Identified: {len(context['all_factors_identified'])}")
			for factor in context['all_factors_identified']:
				print(f"  - {factor}")
		
		print("\n" + "=" * 70 + "\n")


# Global session instance
_session = SessionMemory()


def get_session() -> SessionMemory:
	"""Get the current global session instance."""
	return _session


def reset_global_session() -> None:
	"""Reset the global session."""
	global _session
	_session = SessionMemory()


if __name__ == "__main__":
	# Demo: Session tracking
	session = SessionMemory()
	
	# Add queries
	session.add_query(
		query="Why do customers request refunds?",
		num_evidence_turns=5,
		factors_identified=["Product Defects", "Billing Issues"],
	)
	
	session.add_query(
		query="What are common customer complaints?",
		num_evidence_turns=3,
		factors_identified=["Delivery Delays", "Product Quality"],
	)
	
	# Add evidence
	session.add_evidence(
		transcript_id="t-001",
		turn_range="3-5",
		span_text="Agent: What's wrong?\\nCustomer: Item is defective.",
		similarity_score=0.92,
		source_query="Why do customers request refunds?",
	)
	
	session.add_evidence(
		transcript_id="t-002",
		turn_range="2-4",
		span_text="Agent: Let me help.\\nCustomer: I was overcharged.",
		similarity_score=0.87,
		source_query="Why do customers request refunds?",
	)
	
	# Print summary
	session.print_summary()
	
	# Get context
	context = session.get_context()
	print("Context:", context)
