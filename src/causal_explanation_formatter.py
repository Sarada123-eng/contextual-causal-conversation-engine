"""Deterministic formatter for causal explanations grounded in retrieved evidence."""

from __future__ import annotations

from causal_explainer import Explanation, CausalFactor


def _format_evidence_references(factor: CausalFactor) -> list[str]:
	"""Format evidence references as sorted transcript IDs with turn ranges.

	Args:
		factor: CausalFactor containing evidence turns

	Returns:
		List of reference strings like "t-001 (Turns 3-5)"
	"""
	references = []
	for turn in factor.evidence_turns:
		references.append((turn.transcript_id, turn.turn_range))

	# Sort deterministically by transcript ID then turn range
	references = sorted(references, key=lambda x: (x[0], x[1]))
	return [f"{transcript_id} (Turns {turn_range})" for transcript_id, turn_range in references]


def _format_mechanism(factor: CausalFactor) -> str:
	"""Create a deterministic, evidence-grounded mechanism statement.

	This avoids introducing new facts by referencing retrieved evidence only.
	"""
	if not factor.evidence_turns:
		return "No evidence turns available to describe a mechanism."

	return (
		"In the retrieved turns, this factor is mentioned in the same context "
		"as the outcome request, indicating a consistent association in the evidence."
	)


def format_causal_explanation_report(
	explanation: Explanation,
) -> str:
	"""Format a structured causal explanation report.

	Output format per factor:
	- Cause: Description of the causal factor
	- Mechanism: How the factor leads to the outcome (evidence-grounded)
	- Evidence References: Transcript IDs and turn ranges

	Args:
		explanation: Explanation object with ranked factors and evidence

	Returns:
		Deterministic, structured report string
	"""
	lines: list[str] = []
	lines.append("CAUSAL EXPLANATION REPORT")
	lines.append("=" * 80)
	lines.append(f"Query: {explanation.query}")
	lines.append(f"Total Factors: {len(explanation.factors)}")
	lines.append("")

	for idx, factor in enumerate(explanation.factors, start=1):
		lines.append("-" * 80)
		lines.append(f"FACTOR {idx}: {factor.factor_name}")
		lines.append("-" * 80)
		lines.append("Cause:")
		lines.append(f"  {factor.description}")
		lines.append("")
		lines.append("Mechanism:")
		lines.append(f"  {_format_mechanism(factor)}")
		lines.append("")
		lines.append("Evidence References:")
		references = _format_evidence_references(factor)
		if references:
			for ref in references:
				lines.append(f"  - {ref}")
		else:
			lines.append("  - None")
		lines.append("")

	lines.append("=" * 80)
	lines.append("FACTOR RANKING")
	lines.append("=" * 80)
	lines.append("Rank | Factor Name                      | Freq | Coverage | Similarity")
	lines.append("-----+----------------------------------+------+----------+-----------")
	for rank, factor in enumerate(explanation.factors, start=1):
		lines.append(
			f"{rank:>4} | {factor.factor_name:<32} | {factor.evidence_frequency:>4} "
			f"| {factor.transcript_coverage:>8} | {factor.avg_similarity:>9.3f}"
		)
	lines.append("")

	return "\n".join(lines)
