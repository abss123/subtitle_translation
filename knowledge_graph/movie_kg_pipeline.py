"""Build a movie character knowledge graph and relationship-aware Chinese translations.

The pipeline is designed for the Cornell Movie Dialogs Corpus. It can run in a
mock mode for local validation, or call Gemini for relationship extraction and
line-by-line translation.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, get_args, get_origin

import pandas as pd

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # pragma: no cover - keeps the local demo runnable without pydantic
    class ValidationError(Exception):
        pass

    class _FieldInfo:
        def __init__(self, default: Any = None, default_factory: Any = None):
            self.default = default
            self.default_factory = default_factory

        def make_default(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            return self.default

    def Field(default: Any = None, default_factory: Any = None, **_: Any) -> Any:
        return _FieldInfo(default=default, default_factory=default_factory)

    def _model_annotations(cls: type) -> Dict[str, Any]:
        annotations: Dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            annotations.update(getattr(base, "__annotations__", {}))
        return annotations

    def _coerce_model_value(annotation: Any, value: Any) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in (list, List) and args:
            child_type = args[0]
            if isinstance(value, list):
                return [_coerce_model_value(child_type, item) for item in value]
            return []
        if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, dict):
            return annotation.model_validate(value)
        return value

    def _dump_model_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_dump_model_value(item) for item in value]
        if isinstance(value, dict):
            return {key: _dump_model_value(item) for key, item in value.items()}
        return value

    class BaseModel:
        def __init__(self, **kwargs: Any):
            for name, annotation in _model_annotations(self.__class__).items():
                if name in kwargs:
                    value = kwargs[name]
                else:
                    default = getattr(self.__class__, name, None)
                    value = default.make_default() if isinstance(default, _FieldInfo) else default
                setattr(self, name, _coerce_model_value(annotation, value))

        @classmethod
        def model_validate(cls, payload: Dict[str, Any]) -> "BaseModel":
            if not isinstance(payload, dict):
                raise ValidationError(f"Expected dict for {cls.__name__}")
            return cls(**payload)

        @classmethod
        def model_json_schema(cls) -> Dict[str, Any]:
            return {}

        def model_dump(self) -> Dict[str, Any]:
            return {
                name: _dump_model_value(getattr(self, name))
                for name in _model_annotations(self.__class__)
            }

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from google import genai as google_genai
except Exception:  # pragma: no cover - depends on installed SDK
    google_genai = None

try:
    import google.generativeai as legacy_genai
except Exception:  # pragma: no cover - depends on installed SDK
    legacy_genai = None


SEP = " +++$+++ "

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OUTPUT_DIR = Path("outputs_cornell_gemini")

RELATION_TYPES = [
    "family",
    "romantic_interest",
    "ex_partner",
    "friendship",
    "rivalry",
    "mentor",
    "subordinate",
    "suspicion",
    "alliance",
    "conflict",
    "respect",
    "dependency",
    "unknown",
]

POLARITY_TYPES = ["positive", "negative", "mixed", "uncertain"]


class CharacterEntity(BaseModel):
    """A character entity extracted from a dialogue chunk."""

    canonical_name: str = Field(description="Canonical character name")
    aliases: List[str] = Field(default_factory=list)
    source_speakers: List[str] = Field(default_factory=list)
    description: str = Field(default="")


class RelationshipEdge(BaseModel):
    """A relationship between two characters."""

    source: str
    target: str
    relation_type: str = Field(default="unknown")
    relation_label: str = Field(default="")
    polarity: str = Field(default="uncertain")
    direction: str = Field(default="undirected")
    evidence_turn_ids: List[int] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ChunkExtractionResult(BaseModel):
    """Structured Gemini output for one dialogue chunk."""

    chunk_id: str
    movie_id: str
    movie_title: str
    characters: List[CharacterEntity] = Field(default_factory=list)
    relationships: List[RelationshipEdge] = Field(default_factory=list)
    notes: str = Field(default="")


class TranslatedTurn(BaseModel):
    """A single translated dialogue turn."""

    global_turn_index: int
    line_id: str
    speaker_name: str
    addressee_name: str = Field(default="")
    addressee_confidence: float = Field(default=0.0, ge=0, le=1)
    original_text: str
    relationship_context: str = Field(default="")
    chinese_translation: str


class TranslationBatchResult(BaseModel):
    """Structured Gemini output for a batch of translated turns."""

    movie_id: str
    movie_title: str
    translations: List[TranslatedTurn] = Field(default_factory=list)


class AddresseePrediction(BaseModel):
    """The likely listener for one dialogue turn."""

    global_turn_index: int
    speaker_name: str
    addressee_name: str = Field(default="")
    addressee_confidence: float = Field(default=0.0, ge=0, le=1)
    rationale: str = Field(default="")


class AddresseeBatchResult(BaseModel):
    """Structured Gemini output for addressee detection."""

    movie_id: str
    movie_title: str
    predictions: List[AddresseePrediction] = Field(default_factory=list)


@dataclass
class DialogueTurn:
    global_turn_index: int
    conversation_id: str
    line_id: str
    character_id: str
    speaker_name: str
    text: str


@dataclass
class DialogueChunk:
    chunk_id: str
    movie_id: str
    movie_title: str
    start_turn: int
    end_turn: int
    conversation_ids: List[str]
    turns: List[DialogueTurn] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        rows = []
        for turn in self.turns:
            rows.append(
                f"[turn={turn.global_turn_index}][speaker={turn.speaker_name}]"
                f"[line_id={turn.line_id}] {turn.text}"
            )
        return "\n".join(rows)


class GeminiJsonClient:
    """Small compatibility wrapper for the current and legacy Gemini SDKs."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing.")
        self.model = model
        if google_genai is not None:
            self.backend = "google-genai"
            self.client = google_genai.Client(api_key=api_key)
        elif legacy_genai is not None:
            self.backend = "google-generativeai"
            legacy_genai.configure(api_key=api_key)
            self.client = legacy_genai.GenerativeModel(model)
        else:
            raise ImportError(
                "Install either google-genai or google-generativeai to call Gemini."
            )

    def generate_json(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.backend == "google-genai":
            config: Dict[str, Any] = {
                "temperature": 0,
                "response_mime_type": "application/json",
            }
            if schema:
                config["response_json_schema"] = schema
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            text = getattr(response, "text", None)
        else:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            text = getattr(response, "text", None)

        if not text:
            raise ValueError("Gemini returned an empty response.")
        return extract_json_object(text)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response, including fenced JSON."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def read_split_file(path: Path, expected_cols: int) -> List[List[str]]:
    rows = []
    with path.open("r", encoding="iso-8859-1") as file:
        for line in file:
            parts = line.rstrip("\n").split(SEP)
            if len(parts) < expected_cols:
                parts.extend([""] * (expected_cols - len(parts)))
            rows.append(parts[:expected_cols])
    return rows


def load_cornell_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "lines": data_dir / "movie_lines.txt",
        "conversations": data_dir / "movie_conversations.txt",
        "titles": data_dir / "movie_titles_metadata.txt",
        "characters": data_dir / "movie_characters_metadata.txt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Cornell files: {missing}")

    df_lines = pd.DataFrame(
        read_split_file(required["lines"], 5),
        columns=["line_id", "character_id", "movie_id", "character_name", "text"],
    )
    df_convs = pd.DataFrame(
        read_split_file(required["conversations"], 4),
        columns=["character_id_1", "character_id_2", "movie_id", "utterance_ids"],
    )
    df_titles = pd.DataFrame(
        read_split_file(required["titles"], 6),
        columns=["movie_id", "title", "year", "imdb_rating", "imdb_votes", "genres"],
    )
    df_chars = pd.DataFrame(
        read_split_file(required["characters"], 6),
        columns=["character_id", "character_name", "movie_id", "movie_title", "gender", "position"],
    )
    return df_lines, df_convs, df_titles, df_chars


def normalize_text(value: str) -> str:
    value = str(value).lower().strip()
    return re.sub(r"\s+", " ", value)


def find_movie_by_title(df_titles: pd.DataFrame, title_query: str) -> pd.DataFrame:
    query = normalize_text(title_query)
    titles = df_titles.copy()
    titles["title_norm"] = titles["title"].fillna("").map(normalize_text)
    exact = titles[titles["title_norm"] == query]
    if not exact.empty:
        return exact.drop(columns=["title_norm"])
    contains = titles[titles["title_norm"].str.contains(re.escape(query), na=False)]
    return contains.drop(columns=["title_norm"])


def parse_utterance_id_list(raw: str) -> List[str]:
    try:
        ids = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids]


def build_movie_dialogue(
    movie_id: str,
    df_lines: pd.DataFrame,
    df_convs: pd.DataFrame,
) -> pd.DataFrame:
    lines_sub = df_lines[df_lines["movie_id"] == movie_id].copy()
    convs_sub = df_convs[df_convs["movie_id"] == movie_id].copy()
    line_map = lines_sub.set_index("line_id").to_dict("index")

    turns = []
    for conversation_index, row in convs_sub.reset_index(drop=True).iterrows():
        conversation_id = f"{movie_id}_conv_{conversation_index:05d}"
        for local_turn_index, line_id in enumerate(parse_utterance_id_list(row["utterance_ids"])):
            line = line_map.get(line_id)
            if line is None:
                continue
            turns.append(
                {
                    "global_turn_index": len(turns),
                    "conversation_id": conversation_id,
                    "local_turn_index": local_turn_index,
                    "line_id": line_id,
                    "character_id": line.get("character_id", ""),
                    "speaker_name": str(line.get("character_name", "")).strip(),
                    "text": str(line.get("text", "")).strip(),
                }
            )
    return pd.DataFrame(turns)


def build_turn_objects(movie_turns_df: pd.DataFrame) -> List[DialogueTurn]:
    turns = []
    for _, row in movie_turns_df.iterrows():
        turns.append(
            DialogueTurn(
                global_turn_index=int(row["global_turn_index"]),
                conversation_id=str(row["conversation_id"]),
                line_id=str(row["line_id"]),
                character_id=str(row["character_id"]),
                speaker_name=str(row["speaker_name"]),
                text=str(row["text"]),
            )
        )
    return turns


def chunk_dialogue_turns(
    turns: List[DialogueTurn],
    movie_id: str,
    movie_title: str,
    max_turns_per_chunk: int = 40,
    overlap_turns: int = 8,
    max_approx_chars_per_chunk: int = 6000,
) -> List[DialogueChunk]:
    chunks: List[DialogueChunk] = []
    start = 0
    chunk_index = 0
    while start < len(turns):
        end = min(start + max_turns_per_chunk, len(turns))
        current_turns = turns[start:end]
        while len(current_turns) > 1 and sum(len(t.text) + 80 for t in current_turns) > max_approx_chars_per_chunk:
            current_turns = current_turns[:-1]
            end -= 1
        if not current_turns:
            break
        chunks.append(
            DialogueChunk(
                chunk_id=f"{movie_id}_chunk_{chunk_index:04d}",
                movie_id=movie_id,
                movie_title=movie_title,
                start_turn=current_turns[0].global_turn_index,
                end_turn=current_turns[-1].global_turn_index,
                conversation_ids=sorted({turn.conversation_id for turn in current_turns}),
                turns=current_turns,
            )
        )
        chunk_index += 1
        if end >= len(turns):
            break
        start = max(end - overlap_turns, start + 1)
    return chunks


def build_extraction_prompt(chunk: DialogueChunk) -> str:
    relation_types = ", ".join(RELATION_TYPES)
    polarity_types = ", ".join(POLARITY_TYPES)
    return f"""
You are an expert information extraction system for movie dialogue analysis.

Extract characters and interpersonal relationships from the dialogue chunk.

Rules:
1. Use [speaker=...] as the raw speaker name.
2. Merge obvious speaker aliases into one canonical_name, but do not invent unsupported characters.
3. Extract relationships that are directly supported by dialogue evidence.
4. Use relation_type from this list only: {relation_types}.
5. Use polarity from this list only: {polarity_types}.
6. evidence_turn_ids must use the numeric [turn=...] ids from the chunk.
7. evidence_quotes should be short snippets from the provided dialogue.
8. Return valid JSON only.

Required JSON shape:
{{
  "chunk_id": "{chunk.chunk_id}",
  "movie_id": "{chunk.movie_id}",
  "movie_title": "{chunk.movie_title}",
  "characters": [
    {{
      "canonical_name": "Name",
      "aliases": ["Raw or alternate name"],
      "source_speakers": ["Raw speaker name"],
      "description": "Brief role or behavior supported by this chunk"
    }}
  ],
  "relationships": [
    {{
      "source": "Canonical character name",
      "target": "Canonical character name",
      "relation_type": "friendship",
      "relation_label": "Short natural-language label",
      "polarity": "positive",
      "direction": "directed or undirected",
      "evidence_turn_ids": [1, 2],
      "evidence_quotes": ["short quote"],
      "confidence": 0.75
    }}
  ],
  "notes": ""
}}

Dialogue chunk:
{chunk.to_prompt_text()}
""".strip()


def validate_relation_fields(result: ChunkExtractionResult) -> ChunkExtractionResult:
    for relation in result.relationships:
        if relation.relation_type not in RELATION_TYPES:
            relation.relation_type = "unknown"
        if relation.polarity not in POLARITY_TYPES:
            relation.polarity = "uncertain"
        relation.confidence = max(0.0, min(1.0, float(relation.confidence)))
    return result


def mock_extract_from_chunk(chunk: DialogueChunk) -> ChunkExtractionResult:
    speakers = list(dict.fromkeys(t.speaker_name for t in chunk.turns if t.speaker_name))
    characters = [
        CharacterEntity(
            canonical_name=speaker,
            aliases=[speaker],
            source_speakers=[speaker],
            description="Speaker appearing in this chunk.",
        )
        for speaker in speakers[:10]
    ]
    relationships: List[RelationshipEdge] = []
    for left, right in zip(speakers, speakers[1:]):
        if left == right:
            continue
        relationships.append(
            RelationshipEdge(
                source=left,
                target=right,
                relation_type="unknown",
                relation_label=f"{left} speaks near {right}",
                polarity="uncertain",
                direction="undirected",
                evidence_turn_ids=[chunk.start_turn],
                evidence_quotes=[],
                confidence=0.2,
            )
        )
        if len(relationships) >= 5:
            break
    return ChunkExtractionResult(
        chunk_id=chunk.chunk_id,
        movie_id=chunk.movie_id,
        movie_title=chunk.movie_title,
        characters=characters,
        relationships=relationships,
        notes="Mock output for pipeline validation.",
    )


def extract_chunk_with_gemini(
    chunk: DialogueChunk,
    client: GeminiJsonClient,
    max_retries: int = 4,
    initial_retry_delay_sec: float = 2.0,
    jitter_sec: float = 0.8,
) -> ChunkExtractionResult:
    prompt = build_extraction_prompt(chunk)
    schema = ChunkExtractionResult.model_json_schema()
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            payload = client.generate_json(prompt, schema=schema)
            payload["chunk_id"] = chunk.chunk_id
            payload["movie_id"] = chunk.movie_id
            payload["movie_title"] = chunk.movie_title
            return validate_relation_fields(ChunkExtractionResult.model_validate(payload))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < max_retries:
            delay = initial_retry_delay_sec * (2 ** (attempt - 1)) + random.uniform(0, jitter_sec)
            time.sleep(delay)
    raise RuntimeError(f"Gemini extraction failed for {chunk.chunk_id}: {last_error}")


def normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return re.sub(r"\s+", " ", name)


def choose_best_canonical_name(names: Iterable[str]) -> str:
    clean_names = [str(name).strip() for name in names if str(name).strip()]
    if not clean_names:
        return ""
    return sorted(clean_names, key=lambda value: (-len(value), value.lower()))[0]


def build_character_canonical_map(chunk_results: List[Dict[str, Any]]) -> Dict[str, str]:
    groups: Dict[str, set[str]] = defaultdict(set)
    for result in chunk_results:
        for character in result.get("characters", []):
            names = [character.get("canonical_name", "")]
            names.extend(character.get("aliases", []) or [])
            names.extend(character.get("source_speakers", []) or [])
            normalized_names = [normalize_name(name) for name in names if normalize_name(name)]
            if not normalized_names:
                continue
            anchor = sorted(normalized_names, key=len)[0]
            for name in names:
                if normalize_name(name):
                    groups[anchor].add(str(name).strip())

    canonical_map: Dict[str, str] = {}
    for names in groups.values():
        canonical = choose_best_canonical_name(names)
        for name in names:
            normalized = normalize_name(name)
            if normalized:
                canonical_map[normalized] = canonical
    return canonical_map


def canonicalize_name(name: str, canonical_map: Dict[str, str]) -> str:
    normalized = normalize_name(name)
    return canonical_map.get(normalized, str(name).strip() or normalized)


def merge_relationships(chunk_results: List[Dict[str, Any]], canonical_map: Dict[str, str]) -> pd.DataFrame:
    columns = [
        "source",
        "target",
        "relation_type",
        "relation_label",
        "polarity",
        "direction",
        "evidence_turn_ids",
        "evidence_quotes",
        "chunk_ids",
        "support_count",
        "final_confidence",
    ]
    aggregate: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for result in chunk_results:
        chunk_id = result.get("chunk_id", "")
        for relation in result.get("relationships", []):
            source = canonicalize_name(relation.get("source", ""), canonical_map)
            target = canonicalize_name(relation.get("target", ""), canonical_map)
            if not source or not target or source == target:
                continue
            relation_type = relation.get("relation_type", "unknown")
            direction = relation.get("direction", "undirected")
            key = (source, target, relation_type) if direction == "directed" else tuple(sorted([source, target])) + (relation_type,)

            item = aggregate.setdefault(
                key,
                {
                    "source": source,
                    "target": target,
                    "relation_type": relation_type,
                    "relation_labels": [],
                    "polarities": [],
                    "direction": direction,
                    "evidence_turn_ids": set(),
                    "evidence_quotes": [],
                    "chunk_ids": set(),
                    "confidences": [],
                },
            )
            item["relation_labels"].append(relation.get("relation_label", ""))
            item["polarities"].append(relation.get("polarity", "uncertain"))
            item["evidence_turn_ids"].update(relation.get("evidence_turn_ids", []) or [])
            item["evidence_quotes"].extend(relation.get("evidence_quotes", []) or [])
            item["chunk_ids"].add(chunk_id)
            item["confidences"].append(float(relation.get("confidence", 0.5)))

    rows = []
    for item in aggregate.values():
        labels = [label for label in item["relation_labels"] if label]
        confidence_values = item["confidences"] or [0.5]
        rows.append(
            {
                "source": item["source"],
                "target": item["target"],
                "relation_type": item["relation_type"],
                "relation_label": labels[0] if labels else item["relation_type"],
                "polarity": most_common(item["polarities"]) or "uncertain",
                "direction": item["direction"],
                "evidence_turn_ids": sorted(item["evidence_turn_ids"]),
                "evidence_quotes": dedupe_keep_order(item["evidence_quotes"])[:8],
                "chunk_ids": sorted(item["chunk_ids"]),
                "support_count": len(item["chunk_ids"]),
                "final_confidence": round(sum(confidence_values) / len(confidence_values), 3),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["support_count", "final_confidence"], ascending=False)


def most_common(values: List[str]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def dedupe_keep_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    result = []
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def build_nodes_table(chunk_results: List[Dict[str, Any]], canonical_map: Dict[str, str]) -> pd.DataFrame:
    aliases: Dict[str, set[str]] = defaultdict(set)
    source_speakers: Dict[str, set[str]] = defaultdict(set)
    descriptions: Dict[str, List[str]] = defaultdict(list)

    for result in chunk_results:
        for character in result.get("characters", []):
            names = [character.get("canonical_name", "")]
            names.extend(character.get("aliases", []) or [])
            names.extend(character.get("source_speakers", []) or [])
            canonical_candidates = [canonicalize_name(name, canonical_map) for name in names if normalize_name(name)]
            canonical = choose_best_canonical_name(canonical_candidates)
            if not canonical:
                continue
            aliases[canonical].update(name for name in character.get("aliases", []) or [] if name)
            source_speakers[canonical].update(name for name in character.get("source_speakers", []) or [] if name)
            description = character.get("description", "")
            if description:
                descriptions[canonical].append(description)

    rows = []
    for canonical in sorted(set(aliases) | set(source_speakers) | set(descriptions)):
        rows.append(
            {
                "name": canonical,
                "node_type": "Character",
                "aliases": sorted(aliases[canonical] | {canonical}),
                "source_speakers": sorted(source_speakers[canonical]),
                "description": " ".join(dedupe_keep_order(descriptions[canonical])[:3]),
            }
        )
    return pd.DataFrame(rows)


def relationship_context_by_speaker(edges_df: pd.DataFrame) -> Dict[str, str]:
    return relationship_context_by_participant(pd.DataFrame(), edges_df)


def parse_listish(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [text]
    return []


def build_character_name_lookup(nodes_df: pd.DataFrame) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if nodes_df.empty:
        return lookup
    for _, row in nodes_df.iterrows():
        canonical = str(row.get("name", "")).strip()
        if not canonical:
            continue
        variants = [canonical]
        variants.extend(parse_listish(row.get("aliases", [])))
        variants.extend(parse_listish(row.get("source_speakers", [])))
        for variant in variants:
            normalized = normalize_name(variant)
            if normalized:
                lookup.setdefault(normalized, canonical)
    return lookup


def resolve_character_name(name: str, name_lookup: Dict[str, str]) -> str:
    normalized = normalize_name(name)
    return name_lookup.get(normalized, str(name).strip())


def relationship_context_by_participant(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Dict[str, str]:
    contexts_by_canonical: Dict[str, List[str]] = defaultdict(list)
    if edges_df.empty:
        return {}
    for _, row in edges_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        relation = str(row["relation_type"])
        label = str(row.get("relation_label", ""))
        polarity = str(row.get("polarity", "uncertain"))
        summary = f"{target}: {relation}, {polarity}. {label}".strip()
        contexts_by_canonical[source].append(summary)
        reverse = f"{source}: {relation}, {polarity}. {label}".strip()
        contexts_by_canonical[target].append(reverse)

    contexts: Dict[str, str] = {
        name: " | ".join(items[:8]) for name, items in contexts_by_canonical.items()
    }
    if nodes_df.empty:
        return contexts

    for _, row in nodes_df.iterrows():
        canonical = str(row.get("name", "")).strip()
        context = contexts.get(canonical, "")
        if not context:
            continue
        variants = [canonical]
        variants.extend(parse_listish(row.get("aliases", [])))
        variants.extend(parse_listish(row.get("source_speakers", [])))
        for variant in variants:
            if variant:
                contexts[variant] = context
    return contexts


def relationship_context_for_pair(
    speaker_name: str,
    addressee_name: str,
    name_lookup: Dict[str, str],
    edges_df: pd.DataFrame,
) -> str:
    if edges_df.empty or not addressee_name:
        return ""
    speaker = resolve_character_name(speaker_name, name_lookup)
    addressee = resolve_character_name(addressee_name, name_lookup)
    if not speaker or not addressee:
        return ""

    summaries = []
    for _, row in edges_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        if {source, target} != {speaker, addressee}:
            continue
        relation = str(row.get("relation_type", "unknown"))
        polarity = str(row.get("polarity", "uncertain"))
        label = str(row.get("relation_label", ""))
        direction = str(row.get("direction", "undirected"))
        if source == speaker and target == addressee:
            prefix = f"{speaker} -> {addressee}"
        elif direction == "directed":
            prefix = f"{addressee} -> {speaker}"
        else:
            prefix = f"{speaker} <-> {addressee}"
        summaries.append(f"{prefix}: {relation}, {polarity}. {label}".strip())
    return " | ".join(dedupe_keep_order(summaries)[:4])


def build_addressee_prompt(movie_id: str, movie_title: str, turns: List[DialogueTurn]) -> str:
    speakers = sorted({turn.speaker_name for turn in turns if turn.speaker_name})
    payload = [
        {
            "global_turn_index": turn.global_turn_index,
            "speaker_name": turn.speaker_name,
            "line_id": turn.line_id,
            "text": turn.text,
        }
        for turn in turns
    ]
    return f"""
You are analyzing movie dialogue turns.

For each turn, infer the most likely addressee: the character being spoken to.
Use only this local turn window, speaker alternation, direct names or titles in the line, and conversational flow.
If the addressee is unclear, choose the most likely nearby speaker and use lower confidence.
The addressee_name should usually be one of these speakers: {", ".join(speakers)}.
Return valid JSON only.

Required JSON shape:
{{
  "movie_id": "{movie_id}",
  "movie_title": "{movie_title}",
  "predictions": [
    {{
      "global_turn_index": 0,
      "speaker_name": "Speaker",
      "addressee_name": "Likely listener",
      "addressee_confidence": 0.75,
      "rationale": "Brief evidence from the local turns"
    }}
  ]
}}

Dialogue window:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def detect_addressees_with_gemini(
    movie_id: str,
    movie_title: str,
    turns: List[DialogueTurn],
    client: GeminiJsonClient,
    max_retries: int = 4,
) -> AddresseeBatchResult:
    prompt = build_addressee_prompt(movie_id, movie_title, turns)
    schema = AddresseeBatchResult.model_json_schema()
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            payload = client.generate_json(prompt, schema=schema)
            payload["movie_id"] = movie_id
            payload["movie_title"] = movie_title
            return AddresseeBatchResult.model_validate(payload)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Gemini addressee detection failed: {last_error}")


def detect_addressees_heuristic(
    movie_id: str,
    movie_title: str,
    turns: List[DialogueTurn],
) -> AddresseeBatchResult:
    speakers = [turn.speaker_name for turn in turns]
    predictions = []
    for index, turn in enumerate(turns):
        addressee = ""
        confidence = 0.0
        rationale = "No nearby alternate speaker found."

        text_norm = normalize_text(turn.text)
        for candidate in sorted(set(speakers), key=len, reverse=True):
            if candidate == turn.speaker_name:
                continue
            candidate_norm = normalize_text(candidate)
            if candidate_norm and candidate_norm in text_norm:
                addressee = candidate
                confidence = 0.75
                rationale = "The line directly mentions the addressee name."
                break

        if not addressee:
            for previous in range(index - 1, -1, -1):
                if turns[previous].speaker_name != turn.speaker_name:
                    addressee = turns[previous].speaker_name
                    confidence = 0.55
                    rationale = "Nearest previous different speaker in the local dialogue."
                    break

        if not addressee:
            for following in range(index + 1, len(turns)):
                if turns[following].speaker_name != turn.speaker_name:
                    addressee = turns[following].speaker_name
                    confidence = 0.45
                    rationale = "Nearest following different speaker in the local dialogue."
                    break

        predictions.append(
            AddresseePrediction(
                global_turn_index=turn.global_turn_index,
                speaker_name=turn.speaker_name,
                addressee_name=addressee,
                addressee_confidence=confidence,
                rationale=rationale,
            )
        )
    return AddresseeBatchResult(movie_id=movie_id, movie_title=movie_title, predictions=predictions)


def addressee_map_from_predictions(predictions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    result = {}
    for prediction in predictions:
        try:
            turn_id = int(prediction.get("global_turn_index"))
        except Exception:
            continue
        result[turn_id] = prediction
    return result


def turn_window_for_batch(
    all_turns: List[DialogueTurn],
    batch: List[DialogueTurn],
    context_turns: int,
) -> List[DialogueTurn]:
    if not batch:
        return []
    first = max(0, batch[0].global_turn_index - context_turns)
    last = min(len(all_turns) - 1, batch[-1].global_turn_index + context_turns)
    return all_turns[first : last + 1]


def build_translation_prompt(
    movie_id: str,
    movie_title: str,
    turns: List[DialogueTurn],
    relationship_context: Dict[str, str],
    addressee_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    payload = []
    for turn in turns:
        addressee = (addressee_map or {}).get(turn.global_turn_index, {})
        addressee_name = str(addressee.get("addressee_name", ""))
        addressee_confidence = float(addressee.get("addressee_confidence", 0.0) or 0.0)
        context = relationship_context.get(str(turn.global_turn_index), "")
        if not context:
            context = relationship_context.get(turn.speaker_name, "")
        payload.append(
            {
                "global_turn_index": turn.global_turn_index,
                "line_id": turn.line_id,
                "speaker_name": turn.speaker_name,
                "addressee_name": addressee_name,
                "addressee_confidence": addressee_confidence,
                "relationship_context": context,
                "original_text": turn.text,
            }
        )
    return f"""
You are a professional English-to-Traditional-Chinese subtitle translator.

Translate each movie dialogue line into natural Traditional Chinese.
Use addressee_name and relationship_context to choose tone, pronouns, intimacy, hostility, and politeness.
If addressee_confidence is low, keep the translation natural but avoid overcommitting to intimacy or hostility.
Preserve the meaning and cinematic voice. Do not add explanations.
Return valid JSON only with the required shape.

Required JSON shape:
{{
  "movie_id": "{movie_id}",
  "movie_title": "{movie_title}",
  "translations": [
    {{
      "global_turn_index": 0,
      "line_id": "L000",
      "speaker_name": "Name",
      "addressee_name": "Likely listener",
      "addressee_confidence": 0.75,
      "original_text": "English line",
      "relationship_context": "Short relationship context",
      "chinese_translation": "Traditional Chinese translation"
    }}
  ]
}}

Lines:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def mock_translate_turns(
    movie_id: str,
    movie_title: str,
    turns: List[DialogueTurn],
    relationship_context: Dict[str, str],
    addressee_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> TranslationBatchResult:
    translations = []
    for turn in turns:
        addressee = (addressee_map or {}).get(turn.global_turn_index, {})
        addressee_name = str(addressee.get("addressee_name", ""))
        addressee_confidence = float(addressee.get("addressee_confidence", 0.0) or 0.0)
        context = relationship_context.get(str(turn.global_turn_index), "")
        if not context:
            context = relationship_context.get(turn.speaker_name, "")
        translations.append(
            TranslatedTurn(
            global_turn_index=turn.global_turn_index,
            line_id=turn.line_id,
            speaker_name=turn.speaker_name,
            addressee_name=addressee_name,
            addressee_confidence=addressee_confidence,
            original_text=turn.text,
            relationship_context=context,
            chinese_translation=f"[MOCK zh-TW] {turn.text}",
        )
        )
    return TranslationBatchResult(movie_id=movie_id, movie_title=movie_title, translations=translations)


def translate_turn_batch_with_gemini(
    movie_id: str,
    movie_title: str,
    turns: List[DialogueTurn],
    relationship_context: Dict[str, str],
    client: GeminiJsonClient,
    addressee_map: Optional[Dict[int, Dict[str, Any]]] = None,
    max_retries: int = 4,
) -> TranslationBatchResult:
    prompt = build_translation_prompt(movie_id, movie_title, turns, relationship_context, addressee_map)
    schema = TranslationBatchResult.model_json_schema()
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            payload = client.generate_json(prompt, schema=schema)
            payload["movie_id"] = movie_id
            payload["movie_title"] = movie_title
            return TranslationBatchResult.model_validate(payload)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Gemini translation failed: {last_error}")


def batched(values: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run_pipeline(args: argparse.Namespace) -> Dict[str, Path]:
    if load_dotenv is not None:
        load_dotenv()

    output_dir = Path(args.output_dir)
    chunk_dir = output_dir / "chunk_results"
    log_dir = output_dir / "logs"
    translation_dir = output_dir / "translations"
    for directory in [output_dir, chunk_dir, log_dir, translation_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    df_lines, df_convs, df_titles, df_chars = load_cornell_dataset(Path(args.data_dir))
    matches = find_movie_by_title(df_titles, args.movie_title)
    if matches.empty:
        raise ValueError(f"No Cornell movie title matched: {args.movie_title}")

    selected = matches.iloc[0]
    movie_id = str(selected["movie_id"])
    movie_title = str(selected["title"])
    movie_turns_df = build_movie_dialogue(movie_id, df_lines, df_convs)
    if args.max_turns is not None:
        movie_turns_df = movie_turns_df.head(args.max_turns).copy()

    turns = build_turn_objects(movie_turns_df)
    chunks = chunk_dialogue_turns(
        turns,
        movie_id=movie_id,
        movie_title=movie_title,
        max_turns_per_chunk=args.max_turns_per_chunk,
        overlap_turns=args.chunk_overlap_turns,
        max_approx_chars_per_chunk=args.max_approx_chars_per_chunk,
    )
    selected_chunks = chunks[: args.max_chunks] if args.max_chunks is not None else chunks

    save_json(output_dir / f"{movie_id}_movie_metadata.json", selected.to_dict())
    movie_turns_df.to_csv(output_dir / f"{movie_id}_turns.csv", index=False)

    client = None
    if args.run_llm:
        api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
        client = GeminiJsonClient(api_key=api_key, model=args.model)

    chunk_results: List[Dict[str, Any]] = []
    failed_chunks = []
    for index, chunk in enumerate(selected_chunks, start=1):
        out_path = chunk_dir / f"{chunk.chunk_id}.json"
        if args.skip_existing and out_path.exists():
            with out_path.open("r", encoding="utf-8") as file:
                chunk_results.append(json.load(file))
            continue
        print(f"[{index}/{len(selected_chunks)}] Extracting {chunk.chunk_id}")
        try:
            if args.run_llm and client is not None:
                result = extract_chunk_with_gemini(chunk, client=client, max_retries=args.max_retries)
                time.sleep(args.sleep_between_calls_sec)
            else:
                result = mock_extract_from_chunk(chunk)
            payload = result.model_dump()
            save_json(out_path, payload)
            chunk_results.append(payload)
        except Exception as exc:
            failed_chunks.append({"chunk_id": chunk.chunk_id, "error": str(exc)})
            (log_dir / f"{chunk.chunk_id}.error.txt").write_text(str(exc), encoding="utf-8")

    save_json(output_dir / f"{movie_id}_failed_chunks.json", failed_chunks)

    canonical_map = build_character_canonical_map(chunk_results)
    nodes_df = build_nodes_table(chunk_results, canonical_map)
    edges_df = merge_relationships(chunk_results, canonical_map)

    nodes_csv = output_dir / f"{movie_id}_nodes.csv"
    edges_csv = output_dir / f"{movie_id}_edges.csv"
    nodes_json = output_dir / f"{movie_id}_nodes.json"
    edges_json = output_dir / f"{movie_id}_edges.json"
    nodes_df.to_csv(nodes_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)
    save_json(nodes_json, nodes_df.to_dict(orient="records"))
    save_json(edges_json, edges_df.to_dict(orient="records"))

    translation_csv = output_dir / f"{movie_id}_translations_zh_tw.csv"
    translation_json = output_dir / f"{movie_id}_translations_zh_tw.json"
    addressee_csv = output_dir / f"{movie_id}_addressees.csv"
    addressee_json = output_dir / f"{movie_id}_addressees.json"
    if args.translate:
        speaker_context = relationship_context_by_participant(nodes_df, edges_df)
        name_lookup = build_character_name_lookup(nodes_df)
        start_turn = max(0, args.translation_start_turn)
        translation_turns = turns[start_turn:]
        if args.max_translation_turns is not None:
            translation_turns = translation_turns[: args.max_translation_turns]

        all_addressees: List[Dict[str, Any]] = []
        if args.detect_addressees:
            for batch_index, batch in enumerate(batched(translation_turns, args.translation_batch_size), start=1):
                detection_window = turn_window_for_batch(turns, batch, args.addressee_context_turns)
                batch_turn_ids = {turn.global_turn_index for turn in batch}
                print(
                    f"[addressee batch {batch_index}] Detecting addressees for "
                    f"{len(batch)} turns with {len(detection_window)} local context turns"
                )
                if args.run_llm and client is not None:
                    addressee_result = detect_addressees_with_gemini(
                        movie_id,
                        movie_title,
                        detection_window,
                        client=client,
                        max_retries=args.max_retries,
                    )
                    time.sleep(args.sleep_between_calls_sec)
                else:
                    addressee_result = detect_addressees_heuristic(movie_id, movie_title, detection_window)
                payload = addressee_result.model_dump()
                save_json(translation_dir / f"{movie_id}_addressee_batch_{batch_index:04d}.json", payload)
                all_addressees.extend(
                    prediction
                    for prediction in payload["predictions"]
                    if int(prediction.get("global_turn_index", -1)) in batch_turn_ids
                )
        else:
            all_addressees = detect_addressees_heuristic(movie_id, movie_title, translation_turns).model_dump()["predictions"]

        pd.DataFrame(all_addressees).to_csv(addressee_csv, index=False)
        save_json(addressee_json, all_addressees)

        addressee_map = addressee_map_from_predictions(all_addressees)
        relation_context: Dict[str, str] = dict(speaker_context)
        for turn in translation_turns:
            prediction = addressee_map.get(turn.global_turn_index, {})
            pair_context = relationship_context_for_pair(
                turn.speaker_name,
                str(prediction.get("addressee_name", "")),
                name_lookup,
                edges_df,
            )
            relation_context[str(turn.global_turn_index)] = pair_context or speaker_context.get(turn.speaker_name, "")

        all_translations: List[Dict[str, Any]] = []
        for batch_index, batch in enumerate(batched(translation_turns, args.translation_batch_size), start=1):
            print(f"[translation batch {batch_index}] Translating {len(batch)} turns")
            if args.run_llm and client is not None:
                result = translate_turn_batch_with_gemini(
                    movie_id,
                    movie_title,
                    batch,
                    relation_context,
                    client=client,
                    addressee_map=addressee_map,
                    max_retries=args.max_retries,
                )
                time.sleep(args.sleep_between_calls_sec)
            else:
                result = mock_translate_turns(movie_id, movie_title, batch, relation_context, addressee_map)
            payload = result.model_dump()
            save_json(translation_dir / f"{movie_id}_translation_batch_{batch_index:04d}.json", payload)
            all_translations.extend(payload["translations"])
        pd.DataFrame(all_translations).to_csv(translation_csv, index=False)
        save_json(translation_json, all_translations)

    return {
        "turns_csv": output_dir / f"{movie_id}_turns.csv",
        "nodes_csv": nodes_csv,
        "edges_csv": edges_csv,
        "nodes_json": nodes_json,
        "edges_json": edges_json,
        "addressee_csv": addressee_csv,
        "addressee_json": addressee_json,
        "translation_csv": translation_csv,
        "translation_json": translation_json,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="cornell_movie_dialogs_corpus")
    parser.add_argument("--movie-title", default="Titanic")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-llm", action="store_true", help="Call Gemini for relationship extraction.")
    parser.add_argument("--translate", action="store_true", help="Call Gemini for relationship-aware zh-TW translation.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--translation-start-turn", type=int, default=0)
    parser.add_argument("--max-translation-turns", type=int, default=None)
    parser.add_argument("--max-turns-per-chunk", type=int, default=40)
    parser.add_argument("--chunk-overlap-turns", type=int, default=8)
    parser.add_argument("--max-approx-chars-per-chunk", type=int, default=6000)
    parser.add_argument("--translation-batch-size", type=int, default=20)
    parser.add_argument("--detect-addressees", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--addressee-context-turns", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--sleep-between-calls-sec", type=float, default=4.2)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    outputs = run_pipeline(args)
    print("Saved outputs:")
    for name, path in outputs.items():
        if path.exists():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
