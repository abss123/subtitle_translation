"""
Movie Lookup Module — query the combined dataset as DataFrames.

Usage:
    from lookup_movie import *

    search_movie("alien")              # find movies by name
    get_conversations("pulp fiction")  # speaker-labeled EN dialogue
    get_parallel_df("pulp fiction")    # aligned EN↔ZH subtitle pairs
    get_characters("pulp fiction")     # character metadata
    get_speaker_pairs("pulp fiction")  # who talks to whom
    list_all_movies()                  # full movie list
"""

import os
import json
import pandas as pd

OUTPUT_DIR = "./data/combined"


# ─── Internal helpers ───

def _load_summary():
    with open(os.path.join(OUTPUT_DIR, "summary.json"), 'r') as f:
        return json.load(f)

def _find_movies(query):
    summary = _load_summary()
    q = query.lower().strip()
    exact = [s for s in summary if s['title'].lower() == q]
    if exact:
        return exact
    return [s for s in summary if q in s['title'].lower()]

def _load_movie(entry):
    path = os.path.join(OUTPUT_DIR, f"{entry['cornell_id']}_{entry['imdb_id']}.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _best_match(query):
    results = _find_movies(query)
    if not results:
        print(f"No matches for '{query}'")
        return None
    for r in results:
        if r['title'].lower() == query.lower().strip():
            return r
    return results[0]


# ─── Public API ───

def list_all_movies() -> pd.DataFrame:
    """All movies in the dataset."""
    return pd.DataFrame(_load_summary())

def search_movie(query: str) -> pd.DataFrame:
    """Search movies by name (fuzzy substring match)."""
    results = _find_movies(query)
    if not results:
        print(f"No matches for '{query}'")
        return pd.DataFrame()
    return pd.DataFrame(results)

def get_movie(query: str) -> dict:
    """Load full movie data as a dict."""
    entry = _best_match(query)
    return _load_movie(entry) if entry else {}

def get_conversations(query: str) -> pd.DataFrame:
    """
    Speaker-labeled EN dialogue turns as a flat DataFrame.
    Columns: conversation_id, participant_1, participant_2,
             turn_index, speaker, speaker_id, gender, credit_pos, text
    """
    data = get_movie(query)
    if not data: return pd.DataFrame()
    rows = []
    for i, convo in enumerate(data['conversations']):
        p1, p2 = convo['participants']
        for j, turn in enumerate(convo['turns']):
            rows.append({
                'conversation_id': i, 'participant_1': p1, 'participant_2': p2,
                'turn_index': j, 'line_id': turn['line_id'],
                'speaker': turn['speaker'], 'speaker_id': turn['speaker_id'],
                'gender': turn['gender'], 'credit_pos': turn['credit_pos'],
                'text': turn['text'],
            })
    df = pd.DataFrame(rows)
    print(f"{data['title']} ({data['year']}): "
          f"{len(data['conversations'])} conversations, {len(df)} turns")
    return df

def get_parallel_df(query: str) -> pd.DataFrame:
    """
    Aligned EN↔ZH subtitle pairs as a DataFrame.
    Columns: line_number, en, zh
    """
    data = get_movie(query)
    if not data: return pd.DataFrame()
    pairs = data.get('parallel_subtitles', [])
    if not pairs:
        print(f"No parallel pairs for '{data['title']}'")
        return pd.DataFrame()
    df = pd.DataFrame(pairs)
    df.insert(0, 'line_number', range(1, len(df) + 1))
    print(f"{data['title']} ({data['year']}): {len(df)} aligned EN↔ZH pairs")
    return df

def get_characters(query: str) -> pd.DataFrame:
    """Character metadata. Columns: character_id, name, gender, credit_pos"""
    data = get_movie(query)
    if not data: return pd.DataFrame()
    rows = [{'character_id': cid, **info} for cid, info in data['characters'].items()]
    return pd.DataFrame(rows)

def get_speaker_pairs(query: str) -> pd.DataFrame:
    """Who talks to whom. Columns: speaker_1, speaker_2, conversations, turns"""
    data = get_movie(query)
    if not data: return pd.DataFrame()
    pairs = {}
    for convo in data['conversations']:
        key = tuple(sorted(convo['participants']))
        if key not in pairs:
            pairs[key] = {'conversations': 0, 'turns': 0}
        pairs[key]['conversations'] += 1
        pairs[key]['turns'] += len(convo['turns'])
    rows = [{'speaker_1': k[0], 'speaker_2': k[1], **v} for k, v in pairs.items()]
    return pd.DataFrame(rows).sort_values('turns', ascending=False).reset_index(drop=True)


# ─── CLI ───

if __name__ == "__main__":
    import sys
    query = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input("Movie name: ")
    results = search_movie(query)
    if results.empty:
        exit()
    print(results.to_string(), "\n")
    convos = get_conversations(query)
    if not convos.empty:
        print(convos.head(15).to_string(), "\n")
    parallel = get_parallel_df(query)
    if not parallel.empty:
        print(parallel.head(15).to_string())