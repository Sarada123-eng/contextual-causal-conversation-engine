"""Multi-turn context-aware reasoning engine for conversational analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import pandas as pd

from query_parser import QueryParser, QueryIntent
from session_memory import SessionMemory
from vector_index import semantic_search, load_metadata, load_embedding_model
from evidence_extraction import extract_evidence_from_results
from causal_explainer import generate_explanation, print_explanation
from causal_explanation_formatter import format_causal_explanation_report


@dataclass
class ReasoningResponse:
    """Response from the reasoning engine."""
    query: str
    intent: QueryIntent
    is_followup: bool
    evidence_reused: int
    new_evidence: int
    causal_factors: List[str]
    response_text: str
    evaluation_output: str
    total_transcripts: int


class ContextEngine:
    """Multi-turn context-aware reasoning engine."""
    
    def __init__(
        self,
        index_path: str = "data/faiss_index.bin",
        metadata_path: str = "data/chunk_metadata.csv",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize the reasoning engine.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to chunk metadata CSV
            model_name: SentenceTransformer model name
        """
        self.session = SessionMemory()
        self.parser = QueryParser(session_memory=self.session)
        
        # Store paths for semantic search
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        
        # Load metadata and model for context operations
        self.metadata = load_metadata(metadata_path)
        self.model = load_embedding_model(model_name)
        self.project_root = Path(__file__).resolve().parents[1]
        self.evaluation_output_path = self.project_root / "data" / "evaluation_outputs.csv"
        self.evaluation_index = self._load_evaluation_queries()
        
        print(f"[ContextEngine] Initialized with {len(self.metadata)} chunks")
    
    def process_query(
        self,
        query: str,
        k: int = 5,
        semantic_threshold: float = 0.7,
    ) -> ReasoningResponse:
        """Process a user query with context awareness.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            semantic_threshold: Similarity threshold for semantic matching
        
        Returns:
            ReasoningResponse with grounded answer
        """
        # Step 1: Classify query intent
        classification = self.parser.classify_with_semantic(
            query=query,
            embedding_model=self.model,
            threshold=semantic_threshold,
        )
        
        print(f"\n[Query Intent] {classification.intent.value}")
        print(f"[Context Required] {classification.requires_context}")
        
        # Step 2: Handle comparative queries directly (special case)
        if classification.intent == QueryIntent.COMPARATIVE:
            print(f"[Mode] Comparative Analysis")
            response_text, new_evidence_count = self._handle_comparative(
                query, classification, k
            )
            evaluation_output = self._summarize_response_text(response_text)
            response = ReasoningResponse(
                query=query,
                intent=classification.intent,
                is_followup=len(self.session.queries) > 0,
                evidence_reused=len(self.session.evidence),
                new_evidence=new_evidence_count,
                causal_factors=self._extract_comparison_factors(query),
                response_text=response_text,
                evaluation_output=evaluation_output,
                total_transcripts=len(self.session.transcript_ids_seen),
            )
            self._update_session(query, response, classification)
            self._log_evaluation(query, classification, response.evaluation_output)
            return response
        
        # Step 3: Determine if follow-up for other query types
        is_followup = self._is_followup_query(classification)
        
        # Step 4: Handle based on query type
        if is_followup:
            response = self._process_followup(query, classification, k)
        else:
            response = self._process_new_query(query, classification, k)
        
        # Step 5: Update session memory
        self._update_session(query, response, classification)
        self._log_evaluation(query, classification, response.evaluation_output)
        
        return response

    def _load_evaluation_queries(self) -> dict[str, dict[str, str]]:
        """Load evaluation query metadata for logging alignment.

        Returns:
            Dict keyed by query text with query_id, query_category, remarks
        """
        eval_path = self.project_root / "data" / "evaluation_queries.csv"
        if not eval_path.exists():
            return {}
        try:
            eval_df = pd.read_csv(eval_path)
        except Exception:
            return {}
        index: dict[str, dict[str, str]] = {}
        for _, row in eval_df.iterrows():
            query_text = str(row.get("query", "")).strip()
            if not query_text:
                continue
            index[query_text] = {
                "query_id": str(row.get("query_id", "")).strip(),
                "query_category": str(row.get("query_category", "")).strip(),
                "remarks": str(row.get("remarks", "")).strip(),
            }
        return index

    def _summarize_explanation(self, explanation) -> str:
        """Create a deterministic summary of the causal explanation.

        Args:
            explanation: Explanation object

        Returns:
            Summary string grounded in retrieved evidence metrics
        """
        if not explanation.factors:
            return "No causal factors identified."
        parts = []
        for idx, factor in enumerate(explanation.factors, start=1):
            parts.append(
                f"{idx}. {factor.factor_name} (freq={factor.evidence_frequency}, "
                f"coverage={factor.transcript_coverage}, sim={factor.avg_similarity:.3f})"
            )
        return "Top factors: " + "; ".join(parts)

    def _summarize_response_text(self, text: str, max_len: int = 300) -> str:
        """Summarize non-causal response text deterministically.

        Args:
            text: Raw response text
            max_len: Maximum length of summary

        Returns:
            Condensed summary string
        """
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return compact[:max_len].rstrip() + "..."

    def _log_evaluation(self, query: str, classification, system_output: str) -> None:
        """Append evaluation output for a processed query.

        Args:
            query: User query string
            classification: QueryClassification
            system_output: Summarized causal explanation output
        """
        meta = self.evaluation_index.get(query, {})
        query_id = meta.get("query_id", "UNKNOWN")
        query_category = meta.get("query_category", classification.intent.value)
        remarks = meta.get("remarks", "")

        row = {
            "query_id": query_id,
            "query": query,
            "query_category": query_category,
            "system_output": system_output,
            "remarks": remarks,
        }

        write_header = not self.evaluation_output_path.exists()
        self.evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.evaluation_output_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["query_id", "query", "query_category", "system_output", "remarks"],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    def _is_followup_query(self, classification) -> bool:
        """Determine if query is a follow-up.
        
        Args:
            classification: QueryClassification object
        
        Returns:
            True if follow-up query
        """
        # Follow-up if:
        # - Requires context
        # - References prior factors
        # - Is validation/clarification/evidence request with history
        # - Is comparative query (always treated as follow-up)
        
        has_history = len(self.session.queries) > 0
        
        if not has_history:
            return False
        
        if classification.requires_context:
            return True
        
        if classification.referenced_factors:
            return True
        
        # Comparative queries are always follow-ups if we have history
        if classification.intent == QueryIntent.COMPARATIVE:
            return True
        
        if classification.intent in [
            QueryIntent.FACTOR_VALIDATION,
            QueryIntent.EVIDENCE_REQUEST,
            QueryIntent.CLARIFICATION,
        ]:
            return True
        
        return False
    
    def _process_new_query(
        self,
        query: str,
        classification,
        k: int,
    ) -> ReasoningResponse:
        """Process a new analysis query.
        
        Args:
            query: User query
            classification: QueryClassification
            k: Number of chunks to retrieve
        
        Returns:
            ReasoningResponse
        """
        print(f"[Mode] New Analysis - Performing semantic search...")
        
        # Retrieve relevant chunks
        results = semantic_search(
            query=query,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            model_name=self.model_name,
            k=k,
        )
        
        # Extract evidence turns with improved precision (top 1-2 per chunk)
        evidence_turns = extract_evidence_from_results(
            query=query,
            retrieval_results=results,
            model=self.model,
            top_k=2,  # Improved precision: only top 2 turns per chunk
            context_window=1,
        )
        
        print(f"[Retrieval] Found {len(results)} chunks → {len(evidence_turns)} precision-ranked evidence turns")
        
        # Generate causal explanation
        explanation = generate_explanation(
            query=query,
            evidence_turns=evidence_turns,
            model=self.model,
        )
        evaluation_output = self._summarize_explanation(explanation)
        
        # Add evidence to session
        self.session.add_evidence_batch(evidence_turns, source_query=query)
        
        # Build response text
        response_text = self._format_explanation(explanation)
        
        # Extract transcript IDs
        transcript_ids = set(evidence_turns[i].transcript_id for i in range(len(evidence_turns)))
        
        return ReasoningResponse(
            query=query,
            intent=classification.intent,
            is_followup=False,
            evidence_reused=0,
            new_evidence=len(evidence_turns),
            causal_factors=[f.factor_name for f in explanation.factors],
            response_text=response_text,
            evaluation_output=evaluation_output,
            total_transcripts=len(transcript_ids),
        )
    
    def _process_followup(
        self,
        query: str,
        classification,
        k: int,
    ) -> ReasoningResponse:
        """Process a follow-up query using prior context.
        
        Args:
            query: User query
            classification: QueryClassification
            k: Number of chunks to retrieve
        
        Returns:
            ReasoningResponse
        """
        print(f"[Mode] Follow-up - Checking session memory...")
        
        # Get prior evidence
        prior_evidence = self.session.evidence
        
        if classification.intent == QueryIntent.EVIDENCE_REQUEST:
            # User wants to see specific evidence - filter existing
            response_text = self._handle_evidence_request(query, classification)
            evaluation_output = self._summarize_response_text(response_text)
            
            return ReasoningResponse(
                query=query,
                intent=classification.intent,
                is_followup=True,
                evidence_reused=len(prior_evidence),
                new_evidence=0,
                causal_factors=classification.referenced_factors,
                response_text=response_text,
                evaluation_output=evaluation_output,
                total_transcripts=len(self.session.transcript_ids_seen),
            )
        
        elif classification.intent == QueryIntent.FACTOR_VALIDATION:
            # User wants to validate a factor - use existing evidence
            response_text = self._handle_factor_validation(query, classification)
            evaluation_output = self._summarize_response_text(response_text)
            
            return ReasoningResponse(
                query=query,
                intent=classification.intent,
                is_followup=True,
                evidence_reused=len(prior_evidence),
                new_evidence=0,
                causal_factors=classification.referenced_factors,
                response_text=response_text,
                evaluation_output=evaluation_output,
                total_transcripts=len(self.session.transcript_ids_seen),
            )
        
        else:
            # Refinement query - retrieve new evidence but filter by prior transcripts
            print(f"[Refinement] Filtering by {len(self.session.transcript_ids_seen)} prior transcripts")
            
            results = semantic_search(
                query=query,
                index_path=self.index_path,
                metadata_path=self.metadata_path,
                model_name=self.model_name,
                k=k * 2,  # Retrieve more to filter
            )
            
            # Filter to prior transcript IDs for focused refinement
            filtered_results = results[
                results['transcript_id'].isin(self.session.transcript_ids_seen)
            ].head(k)
            
            if len(filtered_results) == 0:
                # No overlap - fall back to new search
                filtered_results = results.head(k)
                print(f"[Refinement] No overlap - using new search")
            else:
                print(f"[Refinement] Filtered to {len(filtered_results)} chunks from prior context")
            
            # Extract evidence with improved precision
            evidence_turns = extract_evidence_from_results(
                query=query,
                retrieval_results=filtered_results,
                model=self.model,
                top_k=2,  # Precision: only top 2 turns per chunk
                context_window=1,
            )
            
            # Generate explanation
            explanation = generate_explanation(
                query=query,
                evidence_turns=evidence_turns,
                model=self.model,
            )
            evaluation_output = self._summarize_explanation(explanation)
            
            # Add new evidence to session
            self.session.add_evidence_batch(evidence_turns, source_query=query)
            
            response_text = self._format_explanation(explanation)
            
            return ReasoningResponse(
                query=query,
                intent=classification.intent,
                is_followup=True,
                evidence_reused=len(prior_evidence),
                new_evidence=len(evidence_turns),
                causal_factors=[f.factor_name for f in explanation.factors],
                response_text=response_text,
                evaluation_output=evaluation_output,
                total_transcripts=len(self.session.transcript_ids_seen),
            )
    
    def _handle_evidence_request(self, query: str, classification) -> str:
        """Handle evidence request by showing existing spans.
        
        Args:
            query: User query
            classification: QueryClassification
        
        Returns:
            Formatted response text
        """
        if classification.referenced_factors:
            # Show evidence for specific factors
            factor = classification.referenced_factors[0]
            relevant_evidence = [
                e for e in self.session.evidence
                if factor.lower() in e.span_text.lower()
            ][:3]  # Top 3
            
            if relevant_evidence:
                response = f"Evidence for '{factor}':\n\n"
                for i, ev in enumerate(relevant_evidence, 1):
                    response += f"{i}. Transcript {ev.transcript_id} (Turns {ev.turn_range}):\n"
                    response += f"   {ev.span_text[:200]}...\n"
                    response += f"   [Similarity: {ev.similarity_score:.2f}]\n\n"
                return response
            else:
                return f"No evidence found for '{factor}' in current session."
        else:
            # Show recent evidence
            recent = self.session.evidence[-5:]
            response = "Recent evidence:\n\n"
            for i, ev in enumerate(recent, 1):
                response += f"{i}. Transcript {ev.transcript_id} (Turns {ev.turn_range}):\n"
                response += f"   {ev.span_text[:200]}...\n\n"
            return response
    
    def _handle_factor_validation(self, query: str, classification) -> str:
        """Handle factor validation using existing evidence.
        
        Args:
            query: User query
            classification: QueryClassification
        
        Returns:
            Formatted response text
        """
        if not classification.referenced_factors:
            return "No specific factor mentioned for validation."
        
        factor = classification.referenced_factors[0]
        
        # Count evidence supporting this factor
        supporting_evidence = [
            e for e in self.session.evidence
            if factor.lower() in e.span_text.lower()
        ]
        
        avg_score = sum(e.similarity_score for e in supporting_evidence) / len(supporting_evidence) if supporting_evidence else 0
        
        response = f"Validation for '{factor}':\n\n"
        response += f"- Supporting Evidence: {len(supporting_evidence)} spans\n"
        response += f"- Average Similarity: {avg_score:.2f}\n"
        response += f"- Unique Transcripts: {len(set(e.transcript_id for e in supporting_evidence))}\n\n"
        
        if avg_score > 0.8:
            response += "✓ Strong validation - high confidence factor"
        elif avg_score > 0.6:
            response += "~ Moderate validation - consider additional evidence"
        else:
            response += "✗ Weak validation - low confidence"
        
        return response
    
    def _extract_comparison_factors(self, query: str) -> list[str]:
        """Extract factor names from comparative query.
        
        Args:
            query: User query string
        
        Returns:
            List of extracted factor names
        """
        import re
        
        query_lower = query.lower()
        factors = []
        
        # Pattern 1: "X vs Y" or "X versus Y"
        vs_pattern = r'([^?]+?)\s+(?:vs\.?|versus)\s+([^?]+?)(?:\s*\?|$)'
        match = re.search(vs_pattern, query_lower)
        if match:
            factor1 = match.group(1).strip()
            factor2 = match.group(2).strip()
            # Clean up common words
            for prefix in ['compare', 'comparing', 'between']:
                factor1 = factor1.replace(prefix, '').strip()
                factor2 = factor2.replace(prefix, '').strip()
            factors = [factor1.title(), factor2.title()]
            return factors
        
        # Pattern 2: "compare X and Y"
        compare_pattern = r'compare\s+([^?]+?)\s+and\s+([^?]+?)(?:\s*\?|$)'
        match = re.search(compare_pattern, query_lower)
        if match:
            factors = [match.group(1).strip().title(), match.group(2).strip().title()]
            return factors
        
        # Pattern 3: "X or Y" in "which is more common" type questions
        or_pattern = r':\s*([^?]+?)\s+or\s+([^?]+?)\s*\?'
        match = re.search(or_pattern, query_lower)
        if match:
            factors = [match.group(1).strip().title(), match.group(2).strip().title()]
            return factors
        
        # Pattern 4: Look for common factor keywords
        common_factors = [
            'product defects', 'billing issues', 'billing errors', 'delivery delays',
            'delivery problems', 'shipping delays', 'poor quality', 'customer service',
            'wrong product', 'wrong item', 'missing items', 'payment problems',
            'payment issues', 'quality issues', 'service quality', 'return requests'
        ]
        
        for factor in common_factors:
            if factor in query_lower:
                factors.append(factor.title())
        
        return factors[:2]  # Return at most 2 factors
    
    def _handle_comparative(
        self, query: str, classification, k: int
    ) -> tuple[str, int]:
        """Handle comparative query between factors with separate retrieval.
        
        Args:
            query: User query
            classification: QueryClassification
            k: Number of results to retrieve per factor
        
        Returns:
            Tuple of (formatted response text, new evidence count)
        """
        # Extract factors from query if not in classification
        factors = classification.referenced_factors
        if len(factors) < 2:
            factors = self._extract_comparison_factors(query)
        
        if len(factors) < 2:
            return "Please specify two factors to compare (e.g., 'Compare X vs Y').", 0
        
        factor1, factor2 = factors[:2]
        
        print(f"[Comparative] Retrieving evidence for '{factor1}' and '{factor2}'...")
        
        # Retrieve evidence for each factor separately
        evidence1, results1 = self._retrieve_factor_evidence(factor1, k)
        evidence2, results2 = self._retrieve_factor_evidence(factor2, k)
        
        # Add evidence to session
        if evidence1:
            self.session.add_evidence_batch(evidence1, source_query=query)
        if evidence2:
            self.session.add_evidence_batch(evidence2, source_query=query)
        
        # Compute metrics for factor 1
        freq1 = len(evidence1)
        avg_sim1 = sum(e.similarity_score for e in evidence1) / len(evidence1) if evidence1 else 0.0
        transcripts1 = set(e.transcript_id for e in evidence1)
        coverage1 = len(transcripts1)
        
        # Compute metrics for factor 2
        freq2 = len(evidence2)
        avg_sim2 = sum(e.similarity_score for e in evidence2) / len(evidence2) if evidence2 else 0.0
        transcripts2 = set(e.transcript_id for e in evidence2)
        coverage2 = len(transcripts2)
        
        # Compute overlap
        overlap = transcripts1 & transcripts2
        
        # Generate side-by-side comparison
        response = self._format_comparison(
            factor1, factor2,
            freq1, freq2,
            avg_sim1, avg_sim2,
            coverage1, coverage2,
            overlap,
            evidence1, evidence2
        )
        
        total_evidence = len(evidence1) + len(evidence2)
        return response, total_evidence
    
    def _retrieve_factor_evidence(
        self, factor: str, k: int
    ) -> tuple[list, pd.DataFrame]:
        """Retrieve evidence for a specific factor.
        
        Args:
            factor: Factor name to search for
            k: Number of results to retrieve
        
        Returns:
            Tuple of (evidence turns list, results DataFrame)
        """
        # Construct query for this factor
        factor_query = f"Why do customers mention {factor}?"
        
        # Retrieve relevant chunks
        results = semantic_search(
            query=factor_query,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            model_name=self.model_name,
            k=k,
        )
        
        # Extract evidence turns with improved precision
        evidence_turns = extract_evidence_from_results(
            query=factor_query,
            retrieval_results=results,
            model=self.model,
            top_k=2,  # Precision: only top 2 turns per chunk
            context_window=1,
        )
        
        return evidence_turns, results
    
    def _format_comparison(
        self,
        factor1: str,
        factor2: str,
        freq1: int,
        freq2: int,
        avg_sim1: float,
        avg_sim2: float,
        coverage1: int,
        coverage2: int,
        overlap: set,
        evidence1: list,
        evidence2: list,
    ) -> str:
        """Format side-by-side factor comparison.
        
        Args:
            factor1, factor2: Factor names
            freq1, freq2: Evidence frequencies
            avg_sim1, avg_sim2: Average similarity scores
            coverage1, coverage2: Transcript coverage counts
            overlap: Overlapping transcript IDs
            evidence1, evidence2: Evidence turn lists
        
        Returns:
            Formatted comparison text
        """
        response = "\n" + "=" * 80 + "\n"
        response += "COMPARATIVE ANALYSIS\n"
        response += "=" * 80 + "\n\n"
        
        response += f"Comparing: '{factor1}' vs '{factor2}'\n\n"
        
        # Side-by-side metrics
        response += f"{'Metric':<30} {factor1:<25} {factor2:<25}\n"
        response += "-" * 80 + "\n"
        response += f"{'Evidence Frequency':<30} {freq1:<25} {freq2:<25}\n"
        response += f"{'Avg Similarity Score':<30} {avg_sim1:.3f}{' ' * 21} {avg_sim2:.3f}\n"
        response += f"{'Transcript Coverage':<30} {coverage1:<25} {coverage2:<25}\n"
        response += f"{'Overlapping Transcripts':<30} {len(overlap):<25}\n"
        response += "\n"
        
        # Determine which is stronger
        # Composite score: frequency * 0.4 + similarity * 0.6
        max_freq = max(freq1, freq2) if max(freq1, freq2) > 0 else 1
        score1 = (freq1 / max_freq) * 0.4 + avg_sim1 * 0.6
        score2 = (freq2 / max_freq) * 0.4 + avg_sim2 * 0.6
        
        response += "Analysis:\n"
        response += "-" * 80 + "\n"
        
        if score1 > score2:
            diff = ((score1 - score2) / score2 * 100) if score2 > 0 else 100
            response += f"→ '{factor1}' is the STRONGER factor ({diff:.1f}% higher composite score)\n"
            response += f"  Appears in {freq1} evidence spans across {coverage1} transcripts\n"
        elif score2 > score1:
            diff = ((score2 - score1) / score1 * 100) if score1 > 0 else 100
            response += f"→ '{factor2}' is the STRONGER factor ({diff:.1f}% higher composite score)\n"
            response += f"  Appears in {freq2} evidence spans across {coverage2} transcripts\n"
        else:
            response += "→ Both factors show SIMILAR strength\n"
        
        if len(overlap) > 0:
            response += f"\n⚠ {len(overlap)} transcripts mention BOTH factors\n"
        
        # Show sample evidence for each
        response += "\n" + "=" * 80 + "\n"
        response += "SAMPLE EVIDENCE\n"
        response += "=" * 80 + "\n"
        
        if evidence1:
            response += f"\n[{factor1}] Top Evidence:\n"
            for i, ev in enumerate(evidence1[:2], 1):  # Top 2
                response += f"  {i}. Transcript {ev.transcript_id} (Turns {ev.turn_range}) - Score: {ev.similarity_score:.3f}\n"
                response += f"     {ev.central_turn_text[:100]}...\n"
        
        if evidence2:
            response += f"\n[{factor2}] Top Evidence:\n"
            for i, ev in enumerate(evidence2[:2], 1):  # Top 2
                response += f"  {i}. Transcript {ev.transcript_id} (Turns {ev.turn_range}) - Score: {ev.similarity_score:.3f}\n"
                response += f"     {ev.central_turn_text[:100]}...\n"
        
        response += "\n" + "=" * 80 + "\n"
        
        return response
    
    def _format_explanation(self, explanation) -> str:
        """Format explanation as response text.
        
        Args:
            explanation: Explanation object
        
        Returns:
            Formatted text
        """
        return format_causal_explanation_report(explanation)
    
    def _update_session(self, query: str, response: ReasoningResponse, classification):
        """Update session memory after processing.
        
        Args:
            query: User query
            response: ReasoningResponse
            classification: QueryClassification
        """
        # Add query to history
        self.session.add_query(
            query=query,
            num_evidence_turns=response.new_evidence,
            factors_identified=response.causal_factors,
        )
        
        # Note: Evidence is added during _process_new_query and _process_followup directly
        
        print(f"[Session] Updated - Total queries: {len(self.session.queries)}, Evidence: {len(self.session.evidence)}")
    
    def get_session_summary(self):
        """Print current session summary."""
        self.session.print_summary()
    
    def reset(self):
        """Reset the reasoning engine session."""
        self.session.reset_session()
        print("[ContextEngine] Session reset")


if __name__ == "__main__":
    # Demo: Multi-turn context-aware reasoning
    import sys
    
    # Check if data files exist
    try:
        engine = ContextEngine()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have:")
        print("  - data/faiss_index.bin")
        print("  - data/chunk_metadata.csv")
        print("\nRun vector_index.py first to generate these files.")
        sys.exit(1)
    
    # Simulate multi-turn conversation
    queries = [
        "Why do customers request refunds?",
        "Show me examples of Product Defects",
        "Compare Product Defects vs Billing Issues",
        "Are there more cases of delivery problems?",
    ]
    
    print("\n" + "=" * 80)
    print("CONTEXT-AWARE REASONING ENGINE DEMO")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Turn {i}: {query}")
        print("=" * 80)
        
        response = engine.process_query(query, k=3)
        
        print(f"\n[Response]")
        print(f"Intent: {response.intent.value}")
        print(f"Follow-up: {response.is_followup}")
        print(f"Evidence Reused: {response.evidence_reused}")
        print(f"New Evidence: {response.new_evidence}")
        print(f"Factors: {', '.join(response.causal_factors) if response.causal_factors else 'None'}")
        print(f"\n{response.response_text}")
    
    # Show session summary
    engine.get_session_summary()
