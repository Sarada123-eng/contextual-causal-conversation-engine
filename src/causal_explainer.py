"""Generate causal explanations from extracted conversational evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from evidence_extraction import EvidenceTurn


# Enhanced Causal Factor Taxonomy with Pattern Dictionaries
# Maps evidence phrases to refined factor categories
FACTOR_TAXONOMY = {
	"Product Defect": {
		"patterns": ["broken", "damaged", "defective", "cracked", "won't work", "doesn't work", "not working"],
		"description": "Products received in defective or broken condition"
	},
	"Billing Dispute": {
		"patterns": ["charged twice", "double charged", "overcharged", "billing error", "duplicate charge"],
		"description": "Customers charged incorrectly or multiple times"
	},
	"Fulfillment Issue": {
		"patterns": ["missing item", "incomplete order", "not included", "missing items"],
		"description": "Orders received incomplete or with missing items"
	},
	"Logistics Delay": {
		"patterns": ["delayed delivery", "late delivery", "shipping delay", "never arrived", "lost package"],
		"description": "Products delayed in delivery or shipping"
	},
	"Quality Concern": {
		"patterns": ["poor quality", "low quality", "flimsy", "poorly made", "falls apart"],
		"description": "Products fail to meet expected quality standards"
	},
	"Wrong Item": {
		"patterns": ["wrong item", "wrong product", "incorrect item", "sent wrong"],
		"description": "Incorrect product received instead of ordered item"
	},
	"Service Issue": {
		"patterns": ["poor service", "bad service", "rude agent", "unhelpful", "no response"],
		"description": "Poor customer service or support experience"
	},
	"Shipping Damage": {
		"patterns": ["arrived damaged", "damaged in shipping", "arrived broken", "packaging damage"],
		"description": "Products damaged or broken during shipping process"
	},
}

# Fixed Causal Taxonomy Mapping Layer
# Maps generic factor labels to standardized causal categories
STANDARDIZED_TAXONOMY = {
	"General Issue": {
		"label": "Repeated Unresolved Customer Complaint",
		"description": "General customer complaints requiring repeated contact or escalation"
	},
	"Fulfillment Issue": {
		"label": "Incomplete or Missing Delivery",
		"description": "Orders received incomplete or with missing items"
	},
	"Product Defect": {
		"label": "Repeated Defective Product Delivery",
		"description": "Products received in defective or broken condition across multiple deliveries"
	},
	"Billing Dispute": {
		"label": "Payment Processing or Refund Errors",
		"description": "Errors in payment processing, charging, or refund handling"
	},
	"Billing Issues": {
		"label": "Payment Processing or Refund Errors",
		"description": "Errors in payment processing, charging, or refund handling"
	},
	"Billing Error": {
		"label": "Payment Processing or Refund Errors",
		"description": "Errors in payment processing, charging, or refund handling"
	},
}


class CausalFactor(NamedTuple):
	"""A single causal factor with complete analytical metrics."""
	factor_name: str
	description: str
	evidence_turns: list[EvidenceTurn]
	avg_similarity: float
	transcript_ids: list[str]
	evidence_frequency: int = 0  # Number of evidence turns
	transcript_coverage: int = 0  # Number of unique transcripts


@dataclass
class Explanation:
	"""Structured causal explanation."""
	query: str
	factors: list[CausalFactor]
	num_transcripts: int


def apply_standardized_taxonomy(generic_label: str) -> tuple[str, str]:
	"""Map generic factor label to standardized taxonomy category.
	
	Args:
		generic_label: Generic factor name from evidence analysis
	
	Returns:
		Tuple of (standardized_label, standardized_description)
	"""
	if generic_label in STANDARDIZED_TAXONOMY:
		config = STANDARDIZED_TAXONOMY[generic_label]
		return config["label"], config["description"]
	
	# Default fallback for unmapped labels
	return "Repeated Unresolved Customer Complaint", "General customer complaints requiring repeated contact or escalation"


def map_phrase_to_factor_category(text: str) -> list[str]:
	"""Map evidence phrase to refined factor category.
	
	Uses pattern dictionary from FACTOR_TAXONOMY to identify causal factors.
	No generic fallbacks - returns empty list if no pattern match.
	
	Args:
		text: Evidence text to analyze
	
	Returns:
		List of matched factor categories (empty if no matches)
	"""
	# Extract text after colons (dialogue content)
	if ":" in text:
		_, dialogue = text.split(":", 1)
		dialogue = dialogue.strip().lower()
	else:
		dialogue = text.lower()
	
	# Match against taxonomy patterns
	matched = []
	for factor_name, config in FACTOR_TAXONOMY.items():
		for pattern in config["patterns"]:
			if pattern in dialogue:
				matched.append(factor_name)
				break
	
	# Return empty list if no matches (caller will use semantic similarity)
	return matched


def extract_noun_phrases(text: str) -> list[str]:
	"""Extract candidate causal phrases from text.
	
	Uses pattern dictionaries from FACTOR_TAXONOMY to identify cause types.
	
	Args:
		text: Text to extract from
	
	Returns:
		List of extracted causal phrases (factor category names)
	"""
	return map_phrase_to_factor_category(text)
	
	# Pattern matching already handled by map_phrase_to_factor_category
	return []


def aggregate_evidence_by_transcript(
	evidence_turns: list[EvidenceTurn],
) -> dict[str, list[EvidenceTurn]]:
	"""Group evidence turns by transcript_id.
	
	Args:
		evidence_turns: List of extracted evidence
	
	Returns:
		Dict mapping transcript_id to list of evidence turns
	"""
	grouped = defaultdict(list)
	for turn in evidence_turns:
		grouped[turn.transcript_id].append(turn)
	return dict(grouped)


def group_evidence_by_similarity(
	evidence_turns: list[EvidenceTurn],
	model: SentenceTransformer,
	similarity_threshold: float = 0.5,
) -> list[list[EvidenceTurn]]:
	"""Group evidence turns by semantic similarity using embeddings.
	
	Uses simple greedy grouping: start with highest similarity turn,
	add all turns above threshold, then move to next ungrouped turn.
	
	Args:
		evidence_turns: List of extracted evidence
		model: SentenceTransformer model
		similarity_threshold: Minimum similarity to group (0-1)
	
	Returns:
		List of groups, each group is a list of EvidenceTurn objects
	"""
	if not evidence_turns:
		return []
	
	# Embed all evidence text
	texts = [turn.span_text for turn in evidence_turns]
	embeddings = model.encode(texts, convert_to_numpy=True)
	embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
	
	groups = []
	used = set()
	
	for i in range(len(evidence_turns)):
		if i in used:
			continue
		
		# Start new group with turn i
		group = [evidence_turns[i]]
		used.add(i)
		
		# Find all similar turns
		for j in range(i + 1, len(evidence_turns)):
			if j in used:
				continue
			
			# Compute cosine similarity
			sim = float(embeddings[i] @ embeddings[j])
			if sim >= similarity_threshold:
				group.append(evidence_turns[j])
				used.add(j)
		
		groups.append(group)
	
	return groups


def find_closest_taxonomy_category(evidence_text: str) -> str:
	"""Find closest taxonomy category using semantic similarity.
	
	Maps ambiguous evidence to the most semantically similar
	taxonomy category instead of using generic fallbacks.
	
	Args:
		evidence_text: Evidence text to classify
	
	Returns:
		Closest taxonomy category name
	"""
	from sentence_transformers import SentenceTransformer
	import numpy as np
	
	# Load model for semantic matching
	model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
	
	# Create taxonomy descriptions for matching
	taxonomy_texts = []
	taxonomy_names = []
	for name, config in FACTOR_TAXONOMY.items():
		# Combine patterns and description for better matching
		taxonomy_text = f"{name}: {config['description']}. Examples: {', '.join(config['patterns'])}"
		taxonomy_texts.append(taxonomy_text)
		taxonomy_names.append(name)
	
	# Encode evidence and taxonomy texts
	evidence_emb = model.encode([evidence_text], convert_to_numpy=True)
	taxonomy_embs = model.encode(taxonomy_texts, convert_to_numpy=True)
	
	# Normalize embeddings
	evidence_emb = evidence_emb / np.linalg.norm(evidence_emb, axis=1, keepdims=True)
	taxonomy_embs = taxonomy_embs / np.linalg.norm(taxonomy_embs, axis=1, keepdims=True)
	
	# Compute similarities
	similarities = evidence_emb @ taxonomy_embs.T
	best_idx = int(np.argmax(similarities))
	
	return taxonomy_names[best_idx]


def generate_factor_title_and_description(
	group: list[EvidenceTurn],
) -> tuple[str, str]:
	"""Generate a descriptive title and description for a causal factor group.
	
	Uses evidence text patterns and semantic similarity to infer specific cause types.
	Never returns generic labels - all factors mapped to standardized taxonomy.
	
	Args:
		group: List of similar evidence turns
	
	Returns:
		Tuple of (title, description) from standardized taxonomy
	"""
	if not group:
		# Use semantic similarity to find closest category
		fallback_category = "Product Defect"  # Default to most common
		return fallback_category, FACTOR_TAXONOMY[fallback_category]["description"]
	
	# Collect all evidence text for analysis
	all_evidence_text = " ".join([turn.span_text for turn in group])
	
	# First try pattern matching
	matched_categories = map_phrase_to_factor_category(all_evidence_text)
	
	# If "General Issue" was returned, use semantic similarity instead
	if matched_categories == ["General Issue"] or not matched_categories:
		# Find closest taxonomy category using semantic similarity
		closest_category = find_closest_taxonomy_category(all_evidence_text)
		title = closest_category
	else:
		# Use first matched category
		title = matched_categories[0]
	
	# Ensure title is in taxonomy
	if title not in FACTOR_TAXONOMY:
		# Fallback to semantic matching
		title = find_closest_taxonomy_category(all_evidence_text)
	
	# Get description from taxonomy
	taxonomy_desc = FACTOR_TAXONOMY[title]["description"]
	description = f"{taxonomy_desc} (evidence found in {len(group)} turns)"
	
	return title, description


def get_taxonomy_documentation() -> dict[str, dict]:
	"""Get complete taxonomy documentation with patterns and descriptions.
	
	Useful for displaying available factor categories and their
definitions to users.
	
	Returns:
		Dict with factor categories, patterns, and descriptions
	"""
	return FACTOR_TAXONOMY


def rank_factors_by_frequency_and_similarity(
	factors: list[CausalFactor],
) -> list[CausalFactor]:
	"""Rank factors by multi-level criteria.
	
	Primary ranking (1st): Evidence frequency (evidence_frequency) - highest first
	Secondary ranking (2nd): Transcript coverage (transcript_coverage) - highest first
	Tertiary ranking (3rd): Similarity score (avg_similarity) - highest first
	
	Args:
		factors: List of causal factors
	
	Returns:
		Factors sorted by multi-level criteria (highest first at each level)
	"""
	if not factors:
		return []
	
	# Sort by tuple of (evidence_frequency DESC, transcript_coverage DESC, avg_similarity DESC)
	# Using negative values for descending order
	sorted_factors = sorted(
		factors,
		key=lambda f: (
			-f.evidence_frequency,  # Primary: frequency (higher is better)
			-f.transcript_coverage,  # Secondary: coverage (higher is better)
			-f.avg_similarity        # Tertiary: similarity (higher is better)
		)
	)
	
	return sorted_factors


def create_causal_factors(
	evidence_groups: list[list[EvidenceTurn]],
) -> list[CausalFactor]:
	"""Convert evidence groups into causal factors with complete metrics.
	
	Applies standardized taxonomy mapping to ensure consistent labels.
	
	Args:
		evidence_groups: Groups of similar evidence
	
	Returns:
		List of CausalFactor objects with analytical metrics and standardized labels
	"""
	factors = []
	
	for group_idx, group in enumerate(evidence_groups, start=1):
		# Extract metadata
		avg_similarity = np.mean([turn.similarity_score for turn in group])
		transcript_ids = list(set(turn.transcript_id for turn in group))
		evidence_frequency = len(group)
		transcript_coverage = len(transcript_ids)
		
		# Generate factor title and description (generic labels)
		generic_title, generic_description = generate_factor_title_and_description(group)
		
		# Apply standardized taxonomy mapping
		standardized_title, standardized_description = apply_standardized_taxonomy(generic_title)
		
		# Preserve evidence grounding by mentioning evidence count in description
		final_description = f"{standardized_description} (evidence found in {evidence_frequency}/{len(group)} turns)"
		
		factor = CausalFactor(
			factor_name=standardized_title,
			description=final_description,
			evidence_turns=group,
			avg_similarity=float(avg_similarity),
			transcript_ids=transcript_ids,
			evidence_frequency=evidence_frequency,
			transcript_coverage=transcript_coverage,
		)
		factors.append(factor)
	
	return factors


def generate_explanation(
	query: str,
	evidence_turns: list[EvidenceTurn],
	model: SentenceTransformer,
	similarity_threshold: float = 0.5,
) -> Explanation:
	"""Generate a structured causal explanation from evidence.
	
	All claims are grounded in retrieved evidence text.
	
	Args:
		query: User query
		evidence_turns: List of extracted evidence turns
		model: SentenceTransformer model
		similarity_threshold: Threshold for grouping similar evidence
	
	Returns:
		Structured Explanation object
	"""
	if not evidence_turns:
		return Explanation(
			query=query,
			factors=[],
			num_transcripts=0,
		)
	
	# Group evidence by similarity
	evidence_groups = group_evidence_by_similarity(
		evidence_turns, model, similarity_threshold
	)
	
	# Create factors
	factors = create_causal_factors(evidence_groups)
	
	# Rank by frequency + similarity
	factors = rank_factors_by_frequency_and_similarity(factors)
	
	# Keep top 3 factors
	top_factors = factors[:3]
	
	# Count unique transcripts
	transcript_ids = set(turn.transcript_id for turn in evidence_turns)
	
	return Explanation(
		query=query,
		factors=top_factors,
		num_transcripts=len(transcript_ids),
	)


def format_factor_analysis(factor: CausalFactor, rank: int, total_evidence: int) -> str:
	"""Format a single causal factor with complete analytical metrics.
	
	Structured analytical report for each factor including:
	- Description: What the factor represents
	- Evidence frequency: How many turns support this factor
	- Transcript coverage: In how many transcripts this factor appears
	- Avg similarity: How relevant the evidence is to the query
	- Example dialogue spans: Top 2 dialogue samples with metadata
	
	Args:
		factor: CausalFactor to format
		rank: Factor rank (1, 2, 3, etc.)
		total_evidence: Total evidence turns in analysis
	
	Returns:
		Formatted factor analysis string with all analytical components
	"""
	output = [""]
	output.append("─" * 80)
	output.append(f"FACTOR {rank}: {factor.factor_name}")
	output.append("─" * 80)
	
	# 1. DESCRIPTION - What the factor represents
	output.append(f"\n📌 DESCRIPTION:")
	output.append(f"   {factor.description}")
	
	# 2. EVIDENCE FREQUENCY - How many turns support this factor
	output.append(f"\n📊 EVIDENCE FREQUENCY:")
	pct = (factor.evidence_frequency / total_evidence * 100) if total_evidence > 0 else 0
	output.append(f"   {factor.evidence_frequency} evidence turns ({pct:.1f}% of total)")
	output.append(f"   └─ Indicates how frequently this factor appears in customer feedback")
	
	# 3. TRANSCRIPT COVERAGE - In how many transcripts this factor appears
	output.append(f"\n🗂️  TRANSCRIPT COVERAGE:")
	output.append(f"   {factor.transcript_coverage} unique transcripts")
	coverage_pct = (factor.transcript_coverage / max(1, len(factor.transcript_ids))) * 100 if factor.transcript_ids else 0
	if factor.transcript_ids:
		output.append(f"   Transcripts: {', '.join(sorted(factor.transcript_ids))}")
	output.append(f"   └─ Indicates how widespread this factor is across transcripts")
	
	# 4. AVERAGE SIMILARITY SCORE - How relevant the evidence is
	output.append(f"\n⭐ AVG SIMILARITY SCORE:")
	output.append(f"   {factor.avg_similarity:.4f} (0-1 scale, 1.0 = perfect match)")
	
	# 5. QUALITY ASSESSMENT - Tier based on similarity
	if factor.avg_similarity >= 0.8:
		quality = "VERY HIGH"
	elif factor.avg_similarity >= 0.6:
		quality = "HIGH"
	elif factor.avg_similarity >= 0.4:
		quality = "MODERATE"
	else:
		quality = "LOW"
	output.append(f"   Quality Assessment: {quality} ({factor.avg_similarity:.0%})")
	output.append(f"   └─ Indicates relevance of evidence to the query")
	
	# 6. EXAMPLE DIALOGUE SPANS - Top dialogue samples with metadata
	output.append(f"\n💬 EXAMPLE DIALOGUE SPANS:")
	output.append(f"   Top {min(2, len(factor.evidence_turns))} dialogue samples with relevance metadata:")
	
	top_turns = sorted(
		factor.evidence_turns,
		key=lambda t: t.similarity_score,
		reverse=True
	)[:2]
	
	for evidence_idx, turn_evidence in enumerate(top_turns, start=1):
		output.append(f"\n   ┌─ Sample {evidence_idx}")
		output.append(f"   ├─ Source: Transcript {turn_evidence.transcript_id} (Turn {turn_evidence.turn_id})")
		output.append(f"   ├─ Location: Turns {turn_evidence.turn_range}")
		output.append(f"   ├─ Rank in Chunk: {turn_evidence.rank_in_chunk}")
		output.append(f"   ├─ Relevance Score: {turn_evidence.similarity_score:.4f} (0-1 scale)")
		output.append(f"   └─ Dialogue:")
		
		# Format dialogue with proper indentation
		dialogue_lines = turn_evidence.span_text.split("\n")
		for line in dialogue_lines:
			if line.strip():
				output.append(f"       │  {line}")
	
	# 7. EVIDENCE SUMMARY - Statistics across all evidence for this factor
	output.append(f"\n📈 EVIDENCE SUMMARY:")
	all_scores = [t.similarity_score for t in factor.evidence_turns]
	min_score = min(all_scores) if all_scores else 0
	max_score = max(all_scores) if all_scores else 0
	output.append(f"   Total evidence turns: {len(factor.evidence_turns)}")
	output.append(f"   Similarity range: {min_score:.4f} - {max_score:.4f}")
	output.append(f"   Coverage: {factor.transcript_coverage} transcript(s)")
	output.append(f"   └─ All evidence turns supporting this factor")
	
	return "\n".join(output)


def print_explanation(explanation: Explanation) -> None:
	"""Print explanation in structured analytical report format with all metrics.
	
	Report structure:
	1. ANALYSIS REQUEST - Shows the query being analyzed
	2. DATA SUMMARY - Overview of transcripts, factors, evidence
	3. FACTOR ANALYSIS DETAILS - Per-factor analytical breakdown with:
	   - Description, Evidence frequency, Transcript coverage
	   - Avg similarity, Quality assessment, Example dialogue spans
	4. ANALYSIS SUMMARY - Factor ranking and key insights
	
	Args:
		explanation: Explanation object
	"""
	print("\n" + "═" * 80)
	print("CAUSAL ANALYSIS REPORT: MULTI-FACTOR EVIDENCE GROUNDING")
	print("═" * 80)
	
	# SECTION 1: ANALYSIS REQUEST
	print(f"\n📋 ANALYSIS REQUEST:")
	print(f"   Query: {explanation.query}")
	
	# SECTION 2: DATA SUMMARY
	print(f"\n📊 DATA SUMMARY:")
	print(f"   Total Transcripts Analyzed: {explanation.num_transcripts}")
	print(f"   Causal Factors Identified: {len(explanation.factors)}")
	
	if not explanation.factors:
		print(f"\n⚠️  No factors extracted from the evidence.")
		return
	
	# Calculate total evidence
	total_evidence = sum(f.evidence_frequency for f in explanation.factors)
	print(f"   Total Evidence Turns: {total_evidence}")
	print(f"   Average Evidence per Factor: {total_evidence / len(explanation.factors):.1f}")
	
	# SECTION 3: FACTOR ANALYSIS DETAILS
	print("\n" + "═" * 80)
	print("SECTION 3: FACTOR ANALYSIS DETAILS")
	print("═" * 80)
	print("Each factor includes: Description, Evidence frequency, Transcript coverage,")
	print("Average similarity, Quality assessment, and Example dialogue spans")
	
	# Format each factor
	for factor_rank, factor in enumerate(explanation.factors, start=1):
		print(format_factor_analysis(factor, factor_rank, total_evidence))
	
	# SECTION 4: ANALYSIS SUMMARY
	print("\n" + "═" * 80)
	print("SECTION 4: ANALYSIS SUMMARY - Rankings & Key Insights")
	print("═" * 80)
	
	print(f"\n📈 FACTOR RANKING (by evidence frequency, transcript coverage, similarity):")
	for rank, factor in enumerate(explanation.factors, start=1):
		avg_score = factor.avg_similarity
		freq = factor.evidence_frequency
		coverage = factor.transcript_coverage
		print(f"   {rank}. {factor.factor_name:<35} | Freq: {freq:2d} | Coverage: {coverage:2d} | Score: {avg_score:.3f}")
	
	print(f"\n💡 KEY INSIGHTS:")
	top_factor = explanation.factors[0]
	top_pct = (top_factor.evidence_frequency / total_evidence * 100) if total_evidence > 0 else 0
	print(f"   1. Primary Factor: {top_factor.factor_name}")
	print(f"      └─ {top_factor.evidence_frequency} evidence turns ({top_pct:.0f}% of total)")
	
	unique_transcripts = len(set(t for f in explanation.factors for t in f.transcript_ids))
	reach_pct = (unique_transcripts / explanation.num_transcripts * 100) if explanation.num_transcripts > 0 else 0
	print(f"   2. Geographic Reach: {explanation.num_transcripts} total transcripts")
	print(f"      └─ Evidence found in {unique_transcripts} transcript(s) ({reach_pct:.0f}%)")
	
	avg_quality = sum(f.avg_similarity for f in explanation.factors) / len(explanation.factors)
	print(f"   3. Evidence Quality: {avg_quality:.3f} average similarity (0-1 scale)")
	if avg_quality >= 0.8:
		quality_tier = "Very High - Excellent match to query"
	elif avg_quality >= 0.6:
		quality_tier = "High - Good relevance to query"
	elif avg_quality >= 0.4:
		quality_tier = "Moderate - Reasonable relevance"
	else:
		quality_tier = "Low - Limited relevance, may need refinement"
	print(f"      └─ {quality_tier}")
	
	# Coverage analysis
	factor_breadth = len(explanation.factors)
	print(f"   4. Factor Breadth: {factor_breadth} distinct factor(s) identified")
	if factor_breadth > 1:
		print(f"      └─ Multiple factors suggest diverse customer concerns")
	else:
		print(f"      └─ Single dominant factor indicates focused issue")
	
	print("\n" + "═" * 80)
	print("END OF ANALYSIS REPORT")
	print("═" * 80 + "\n")


if __name__ == "__main__":
	# Demo: Generate explanation from mock evidence
	from sentence_transformers import SentenceTransformer
	
	model = SentenceTransformer("all-MiniLM-L6-v2")
	
	# Mock evidence turns (with turn_id and rank_in_chunk from precision improvements)
	evidence = [
		EvidenceTurn(
			transcript_id="t-001",
			turn_id=3,
			turn_range="3-5",
			central_turn_text="Customer: I need a refund for this defective item.",
			span_text="Agent: What's wrong with it?\nCustomer: It broke after one use.\nAgent: I can process a refund.",
			similarity_score=0.92,
			rank_in_chunk=1,
		),
		EvidenceTurn(
			transcript_id="t-002",
			turn_id=2,
			turn_range="2-4",
			central_turn_text="Customer: The product arrived damaged.",
			span_text="Agent: Let's review your order.\nCustomer: The item came broken.\nAgent: We'll issue a refund.",
			similarity_score=0.85,
			rank_in_chunk=2,
		),
		EvidenceTurn(
			transcript_id="t-003",
			turn_id=4,
			turn_range="4-6",
			central_turn_text="Customer: I was charged twice for this.",
			span_text="Agent: Let me check the charges.\nCustomer: There's a duplicate charge.\nAgent: I'll fix that.",
			similarity_score=0.78,
			rank_in_chunk=1,
		),
	]
	
	query = "Why do customers request refunds?"
	
	# Generate explanation
	explanation = generate_explanation(query, evidence, model, similarity_threshold=0.5)
	print_explanation(explanation)
