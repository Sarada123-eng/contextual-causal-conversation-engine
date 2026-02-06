"""Utilities for flattening conversational transcripts and building turn windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import json

import pandas as pd


@dataclass(frozen=True)
class TurnRecord:
	transcript_id: str
	domain: str
	intent: str
	reason_for_call: Optional[str]
	turn_id: int
	speaker: str
	text: str


@dataclass(frozen=True)
class ChunkRecord:
	transcript_id: str
	domain: str
	intent: str
	reason_for_call: Optional[str]
	chunk_id: int
	start_turn_id: int
	end_turn_id: int
	turns: List[Dict[str, Any]]
	chunk_text: str


def load_transcripts(json_path: str | Path) -> List[Dict[str, Any]]:
	"""Load the JSON file and return the list of transcripts.

	Expected top-level structure:
	{
		"transcripts": [
			{
				"transcript_id": "...",
				"domain": "...",
				"intent": "...",
				"reason_for_call": "...",
				"conversation": [
					{"speaker": "...", "text": "..."},
					...
				]
			},
			...
		]
	}
	"""
	json_path = Path(json_path)
	with json_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	transcripts = payload.get("transcripts")
	if transcripts is None or not isinstance(transcripts, list):
		raise ValueError("JSON must contain a top-level 'transcripts' list.")
	return transcripts


def flatten_turns(transcripts: Sequence[Dict[str, Any]]) -> List[TurnRecord]:
	"""Flatten transcripts into turn-level records with turn ordering."""
	flattened: List[TurnRecord] = []
	for transcript in transcripts:
		transcript_id = str(transcript.get("transcript_id", ""))
		domain = str(transcript.get("domain", ""))
		intent = str(transcript.get("intent", ""))
		reason_for_call = transcript.get("reason_for_call")

		conversation = transcript.get("conversation") or []
		if not isinstance(conversation, list):
			raise ValueError(
				"Each transcript must include a 'conversation' list of turns."
			)

		for index, turn in enumerate(conversation, start=1):
			speaker = str(turn.get("speaker", ""))
			text = str(turn.get("text", ""))
			flattened.append(
				TurnRecord(
					transcript_id=transcript_id,
					domain=domain,
					intent=intent,
					reason_for_call=reason_for_call,
					turn_id=index,
					speaker=speaker,
					text=text,
				)
			)
	return flattened


def _window_indices(total: int, window_size: int, overlap: int) -> Iterator[Tuple[int, int]]:
	if window_size <= 0:
		raise ValueError("window_size must be positive.")
	if overlap < 0:
		raise ValueError("overlap must be non-negative.")
	if overlap >= window_size:
		raise ValueError("overlap must be smaller than window_size.")

	step = window_size - overlap
	start = 0
	while start < total:
		end = min(start + window_size, total)
		yield start, end
		if end == total:
			break
		start += step


def format_turn(speaker: str, text: str) -> str:
	"""Format a dialogue turn as "Speaker: Text" with capitalized speaker."""
	speaker_key = str(speaker).strip().lower()
	if speaker_key == "agent":
		speaker_label = "Agent"
	else:
		speaker_label = "Customer"
	return f"{speaker_label}: {text}"


def build_chunks(
	turn_records: Sequence[TurnRecord],
	window_size: int = 5,
	overlap: int = 2,
) -> List[ChunkRecord]:
	"""Create sliding window chunks per transcript."""
	chunks: List[ChunkRecord] = []
	turns_by_transcript: Dict[str, List[TurnRecord]] = {}
	for turn in turn_records:
		turns_by_transcript.setdefault(turn.transcript_id, []).append(turn)

	for transcript_id, turns in turns_by_transcript.items():
		turns_sorted = sorted(turns, key=lambda t: t.turn_id)
		for idx, (start, end) in enumerate(
			_window_indices(len(turns_sorted), window_size, overlap), start=1
		):
			window = turns_sorted[start:end]
			if not window:
				continue
			chunk_text_parts: List[str] = []
			for t in window:
				chunk_text_parts.append(format_turn(t.speaker, t.text))
			chunk_text = "\n".join(chunk_text_parts)

			chunks.append(
				ChunkRecord(
					transcript_id=transcript_id,
					domain=window[0].domain,
					intent=window[0].intent,
					reason_for_call=window[0].reason_for_call,
					chunk_id=idx,
					start_turn_id=window[0].turn_id,
					end_turn_id=window[-1].turn_id,
					turns=[
						{
							"turn_id": t.turn_id,
							"speaker": t.speaker,
							"text": t.text,
						}
						for t in window
					],
					chunk_text=chunk_text,
				)
			)

	return chunks


def turns_to_dataframe(turn_records: Sequence[TurnRecord]) -> pd.DataFrame:
	"""Convert turn records to a pandas DataFrame."""
	return pd.DataFrame([turn.__dict__ for turn in turn_records])


def chunks_to_dataframe(chunk_records: Sequence[ChunkRecord]) -> pd.DataFrame:
	"""Convert chunk records to a pandas DataFrame."""
	return pd.DataFrame([chunk.__dict__ for chunk in chunk_records])


def chunks_to_csv_dataframe(chunk_records: Sequence[ChunkRecord]) -> pd.DataFrame:
	"""Convert chunk records to a DataFrame with CSV-ready columns."""
	rows = [
		{
			"chunk_id": chunk.chunk_id,
			"transcript_id": chunk.transcript_id,
			"start_turn": chunk.start_turn_id,
			"end_turn": chunk.end_turn_id,
			"chunk_text": chunk.chunk_text,
			"domain": chunk.domain,
			"intent": chunk.intent,
		}
		for chunk in chunk_records
	]
	return pd.DataFrame(rows)


def json_to_chunk_csv(
	json_path: str | Path,
	output_csv: str | Path = Path("data") / "conversation_chunks.csv",
	window_size: int = 5,
	overlap: int = 2,
	print_samples: bool = True,
	sample_size: int = 2,
) -> pd.DataFrame:
	"""Pipeline: load JSON, flatten turns, build chunks, save chunks CSV.

	Returns the chunks DataFrame.
	"""
	transcripts = load_transcripts(json_path)
	turn_records = flatten_turns(transcripts)
	chunk_records = build_chunks(turn_records, window_size=window_size, overlap=overlap)
	if print_samples:
		print_sample_chunks(chunk_records, sample_size=sample_size)
	output_csv = Path(output_csv)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	chunk_df = chunks_to_csv_dataframe(chunk_records)
	chunk_df.to_csv(output_csv, index=False)
	print(f"✓ CSV saved: {output_csv} ({len(chunk_df)} chunks)")
	return chunk_df


def validate_chunk_integrity(
	turn_records: Sequence[TurnRecord],
	chunk_records: Sequence[ChunkRecord],
) -> None:
	"""Assert that chunking preserves transcript_id and turn ranges.

	Raises AssertionError if any integrity rule is violated.
	"""
	turns_by_transcript: Dict[str, List[TurnRecord]] = {}
	for turn in turn_records:
		turns_by_transcript.setdefault(turn.transcript_id, []).append(turn)

	turns_index: Dict[str, Dict[int, TurnRecord]] = {}
	for transcript_id, turns in turns_by_transcript.items():
		turns_sorted = sorted(turns, key=lambda t: t.turn_id)
		turns_index[transcript_id] = {turn.turn_id: turn for turn in turns_sorted}

	for chunk in chunk_records:
		assert chunk.transcript_id in turns_index, (
			"Chunk transcript_id not found in turns: "
			f"{chunk.transcript_id}"
		)
		turn_lookup = turns_index[chunk.transcript_id]
		turn_ids = [turn["turn_id"] for turn in chunk.turns]
		assert turn_ids, "Chunk has no turns."
		assert chunk.start_turn_id == min(turn_ids), (
			"start_turn_id does not match earliest turn_id in chunk."
		)
		assert chunk.end_turn_id == max(turn_ids), (
			"end_turn_id does not match latest turn_id in chunk."
		)
		assert chunk.start_turn_id <= chunk.end_turn_id, (
			"start_turn_id must be <= end_turn_id."
		)
		for turn_id in turn_ids:
			assert turn_id in turn_lookup, (
				"Chunk references turn_id not present in transcript: "
				f"{turn_id}"
			)
			assert turn_lookup[turn_id].transcript_id == chunk.transcript_id, (
				"Turn transcript_id mismatch within chunk."
			)


def print_sample_chunks(
	chunk_records: Sequence[ChunkRecord],
	sample_size: int = 2,
) -> None:
	"""Print a few chunk samples for inspection."""
	if sample_size <= 0:
		return
	for chunk in list(chunk_records)[:sample_size]:
		print(
			"\n".join(
				[
					f"Chunk {chunk.chunk_id} | transcript_id={chunk.transcript_id}",
					f"turns={chunk.start_turn_id}-{chunk.end_turn_id}",
					chunk.chunk_text,
				]
			)
		)
		print("-" * 40)


def _self_test() -> None:
	transcripts = [
		{
			"transcript_id": "t-001",
			"domain": "billing",
			"intent": "refund",
			"reason_for_call": "incorrect charge",
			"conversation": [
				{"speaker": "agent", "text": "Hello"},
				{"speaker": "customer", "text": "Hi"},
				{"speaker": "agent", "text": "How can I help?"},
				{"speaker": "customer", "text": "Refund issue"},
				{"speaker": "agent", "text": "Let me check"},
				{"speaker": "customer", "text": "Thanks"},
			],
		}
	]
	turn_records = flatten_turns(transcripts)
	chunk_records = build_chunks(turn_records, window_size=5, overlap=2)
	validate_chunk_integrity(turn_records, chunk_records)
	print_sample_chunks(chunk_records, sample_size=2)


__all__ = [
	"TurnRecord",
	"ChunkRecord",
	"load_transcripts",
	"flatten_turns",
	"build_chunks",
	"turns_to_dataframe",
	"chunks_to_dataframe",
	"chunks_to_csv_dataframe",
	"json_to_chunk_csv",
	"validate_chunk_integrity",
	"print_sample_chunks",
]


if __name__ == "__main__":
	json_input_path = "data/Conversational_Transcript_Dataset.json"

	json_to_chunk_csv(
		json_path=json_input_path,
		output_csv="data/conversation_chunks.csv",
		window_size=5,
		overlap=2
	)
