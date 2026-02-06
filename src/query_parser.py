"""Classify user queries into analytical intent types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import re


class QueryIntent(Enum):
	"""Query intent categories."""
	NEW_CAUSAL = "new_causal_query"
	FACTOR_VALIDATION = "factor_validation_query"
	EVIDENCE_REQUEST = "evidence_request"
	COMPARATIVE = "comparative_query"
	CLARIFICATION = "clarification_query"


@dataclass
class QueryClassification:
	"""Classification result for a user query."""
	query: str
	intent: QueryIntent
	confidence: float
	requires_context: bool
	matched_keywords: List[str]
	referenced_factors: List[str]


class QueryParser:
	"""Classifies user queries into analytical intent types."""
	
	# Keyword patterns for each intent type
	CAUSAL_KEYWORDS = [
		r'\bwhy\b',
		r'\bcause\b',
		r'\breason\b',
		r'\blead to\b',
		r'\bresult in\b',
		r'\btrigger\b',
		r'\bexplain\b',
		r'\bcontribute\b',
		r'\bfactor\b',
		r'\bdriver\b',
	]
	
	VALIDATION_KEYWORDS = [
		r'\bverify\b',
		r'\bconfirm\b',
		r'\bvalidate\b',
		r'\bcheck if\b',
		r'\bis it true\b',
		r'\bprove\b',
		r'\bsupport\b',
		r'\bevidence for\b',
		r'\bhow strong\b',
		r'\bhow valid\b',
	]
	
	EVIDENCE_KEYWORDS = [
		r'\bshow me\b',
		r'\bexample\b',
		r'\btranscript\b',
		r'\bconversation\b',
		r'\bevidence\b',
		r'\bspecific case\b',
		r'\binstance\b',
		r'\bdetails\b',
		r'\bsource\b',
		r'\bproof\b',
	]
	
	COMPARATIVE_KEYWORDS = [
		r'\bcompare\b',
		r'\bdifference\b',
		r'\bvs\b',
		r'\bversus\b',
		r'\bbetter\b',
		r'\bworse\b',
		r'\bmore than\b',
		r'\bless than\b',
		r'\bcontrast\b',
		r'\brelative to\b',
	]
	
	CLARIFICATION_KEYWORDS = [
		r'\bwhat do you mean\b',
		r'\bcan you clarify\b',
		r'\bexplain that\b',
		r'\bmore detail\b',
		r'\belaborate\b',
		r'\bspecifically\b',
		r'\bwhich one\b',
		r'\bwhat is\b',
		r'\bdefine\b',
	]
	
	def __init__(self, session_memory=None):
		"""Initialize query parser.
		
		Args:
			session_memory: SessionMemory instance for context detection
		"""
		self.session_memory = session_memory
	
	def classify(self, query: str) -> QueryClassification:
		"""Classify a query into an intent type.
		
		Args:
			query: User query string
		
		Returns:
			QueryClassification with intent and metadata
		"""
		query_lower = query.lower()
		
		# Score each intent type
		scores = {
			QueryIntent.NEW_CAUSAL: self._score_keywords(query_lower, self.CAUSAL_KEYWORDS),
			QueryIntent.FACTOR_VALIDATION: self._score_keywords(query_lower, self.VALIDATION_KEYWORDS),
			QueryIntent.EVIDENCE_REQUEST: self._score_keywords(query_lower, self.EVIDENCE_KEYWORDS),
			QueryIntent.COMPARATIVE: self._score_keywords(query_lower, self.COMPARATIVE_KEYWORDS),
			QueryIntent.CLARIFICATION: self._score_keywords(query_lower, self.CLARIFICATION_KEYWORDS),
		}
		
		# Find best match
		best_intent = max(scores, key=scores.get)
		confidence = scores[best_intent]
		
		# If no strong match, default to new causal query
		if confidence == 0:
			best_intent = QueryIntent.NEW_CAUSAL
			confidence = 0.3
		
		# Detect context dependency
		requires_context = self._requires_context(query_lower)
		
		# Extract matched keywords
		matched = self._extract_matched_keywords(query_lower, best_intent)
		
		# Extract referenced factors
		referenced_factors = self._extract_referenced_factors(query)
		
		return QueryClassification(
			query=query,
			intent=best_intent,
			confidence=confidence,
			requires_context=requires_context,
			matched_keywords=matched,
			referenced_factors=referenced_factors,
		)
	
	def _score_keywords(self, query: str, keywords: List[str]) -> float:
		"""Score query against keyword patterns.
		
		Args:
			query: Lowercased query string
			keywords: List of regex patterns
		
		Returns:
			Score between 0 and 1
		"""
		matches = 0
		for pattern in keywords:
			if re.search(pattern, query):
				matches += 1
		
		# Normalize by pattern count
		return matches / len(keywords) if keywords else 0
	
	def _extract_matched_keywords(self, query: str, intent: QueryIntent) -> List[str]:
		"""Extract which keywords matched for the given intent.
		
		Args:
			query: Lowercased query string
			intent: Classified intent type
		
		Returns:
			List of matched keyword patterns
		"""
		keyword_map = {
			QueryIntent.NEW_CAUSAL: self.CAUSAL_KEYWORDS,
			QueryIntent.FACTOR_VALIDATION: self.VALIDATION_KEYWORDS,
			QueryIntent.EVIDENCE_REQUEST: self.EVIDENCE_KEYWORDS,
			QueryIntent.COMPARATIVE: self.COMPARATIVE_KEYWORDS,
			QueryIntent.CLARIFICATION: self.CLARIFICATION_KEYWORDS,
		}
		
		patterns = keyword_map.get(intent, [])
		matched = []
		
		for pattern in patterns:
			if re.search(pattern, query):
				# Remove regex markers for readability
				keyword = pattern.replace(r'\b', '').replace('\\', '')
				matched.append(keyword)
		
		return matched
	
	def _requires_context(self, query: str) -> bool:
		"""Detect if query depends on prior context.
		
		Args:
			query: Lowercased query string
		
		Returns:
			True if query references prior context
		"""
		context_indicators = [
			r'\bthat\b',
			r'\bthis\b',
			r'\bthose\b',
			r'\bthese\b',
			r'\bprevious\b',
			r'\bprior\b',
			r'\bearlier\b',
			r'\babove\b',
			r'\bsame\b',
			r'\bmore\b',
			r'\balso\b',
			r'\badditional\b',
			r'\banother\b',
			r'\bother\b',
		]
		
		for pattern in context_indicators:
			if re.search(pattern, query):
				return True
		
		return False
	
	def _extract_referenced_factors(self, query: str) -> List[str]:
		"""Extract causal factors mentioned in query.
		
		Args:
			query: Original query string
		
		Returns:
			List of factor names found in session memory
		"""
		if not self.session_memory:
			return []
		
		context = self.session_memory.get_context()
		all_factors = context.get('all_factors_identified', [])
		
		referenced = []
		query_lower = query.lower()
		
		for factor in all_factors:
			if factor.lower() in query_lower:
				referenced.append(factor)
		
		return referenced
	
	def classify_with_semantic(
		self,
		query: str,
		embedding_model=None,
		threshold: float = 0.7,
	) -> QueryClassification:
		"""Classify using semantic similarity in addition to keywords.
		
		Args:
			query: User query string
			embedding_model: SentenceTransformer model for semantic matching
			threshold: Similarity threshold for semantic matches
		
		Returns:
			Enhanced QueryClassification
		"""
		# Start with keyword classification
		classification = self.classify(query)
		
		# If embedding model provided, enhance with semantic matching
		if embedding_model and self.session_memory:
			query_history = self.session_memory.get_query_history()
			
			if query_history:
				# Encode current query and past queries
				import numpy as np
				
				current_embedding = embedding_model.encode([query])[0]
				past_queries = [q['query'] for q in query_history]
				past_embeddings = embedding_model.encode(past_queries)
				
				# Compute similarities
				similarities = np.dot(past_embeddings, current_embedding)
				max_similarity = float(np.max(similarities))
				
				# If high similarity, likely requires context
				if max_similarity > threshold:
					classification.requires_context = True
		
		return classification


if __name__ == "__main__":
	# Demo: Query classification
	from session_memory import SessionMemory
	
	# Create session with some factors
	session = SessionMemory()
	session.add_query(
		query="Why do customers request refunds?",
		factors_identified=["Product Defects", "Billing Issues", "Delivery Delays"],
	)
	
	# Initialize parser
	parser = QueryParser(session_memory=session)
	
	# Test queries
	test_queries = [
		"Why do customers complain about shipping?",
		"Can you validate the Product Defects factor?",
		"Show me examples of billing issues",
		"Compare Product Defects vs Delivery Delays",
		"What do you mean by that factor?",
		"Are there more cases of Product Defects?",
		"Explain the reason customers are unhappy",
		"Verify those claims with evidence",
	]
	
	print("\n" + "=" * 70)
	print("QUERY CLASSIFICATION DEMO")
	print("=" * 70 + "\n")
	
	for query in test_queries:
		classification = parser.classify(query)
		
		print(f"Query: {classification.query}")
		print(f"Intent: {classification.intent.value}")
		print(f"Confidence: {classification.confidence:.2f}")
		print(f"Requires Context: {classification.requires_context}")
		
		if classification.matched_keywords:
			print(f"Matched Keywords: {', '.join(classification.matched_keywords)}")
		
		if classification.referenced_factors:
			print(f"Referenced Factors: {', '.join(classification.referenced_factors)}")
		
		print("-" * 70)
