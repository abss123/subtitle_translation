"""
Step 3: Build Combined Dataset
================================
Requires: Steps 1 and 2 completed.

Merges:
  - Cornell screenplay dialogue (with speaker labels, character metadata)
  - OPUS aligned EN↔ZH subtitle pairs

Into: one JSON file per movie at data/combined/{cornell_id}_{imdb_id}.json

Run: python step3_build_dataset.py
"""

import os
import json

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = "./data"
CORNELL_DIR = os.path.join(DATA_DIR, "cornell")
OPUS_DIR = os.path.join(DATA_DIR, "opus")
OUTPUT_DIR = os.path.join(DATA_DIR, "combined")

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load Cornell parsed data
    parsed_path = os.path.join(CORNELL_DIR, "parsed_cornell.json")
    if not os.path.exists(parsed_path):
        print(f"ERROR: {parsed_path} not found. Run step1_cornell_and_imdb.py first!")
        return
    
    print("Loading Cornell data...")
    with open(parsed_path, 'r', encoding='utf-8') as f:
        cornell = json.load(f)
    
    movies = cornell['movies']
    characters = cornell['characters']
    lines = cornell['lines']
    conversations = cornell['conversations']
    
    # Load IMDB mapping
    mapping_path = os.path.join(CORNELL_DIR, "imdb_mapping.json")
    with open(mapping_path, 'r') as f:
        imdb_mapping = json.load(f)
    
    # Load parallel subtitle pairs (may not exist if step 2 wasn't run)
    parallel_path = os.path.join(OPUS_DIR, "parallel_en_zh.json")
    parallel_data = {}
    if os.path.exists(parallel_path):
        with open(parallel_path, 'r', encoding='utf-8') as f:
            parallel_data = json.load(f)
        print(f"Loaded parallel data: {len(parallel_data)} movies")
    else:
        print(f"Warning: {parallel_path} not found. Run step2_parallel_subtitles.py")
        print("Proceeding with Cornell data only (no ZH subtitles).\n")
    
    # Build combined dataset
    print("Building combined dataset...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []
    
    for movie_id, info in movies.items():
        imdb_info = imdb_mapping.get(movie_id)
        if not imdb_info:
            continue
        
        imdb_id = imdb_info["imdb_id"]
        
        # Characters for this movie
        movie_chars = {cid: ci for cid, ci in characters.items()
                       if ci["movieID"] == movie_id}
        
        # Conversations for this movie
        movie_convos = []
        for convo in conversations:
            if convo["movieID"] != movie_id:
                continue
            turns = []
            for lid in convo["line_ids"]:
                if lid not in lines:
                    continue
                li = lines[lid]
                ci = characters.get(li["charID"], {})
                turns.append({
                    "line_id": lid,
                    "speaker": li["char_name"],
                    "speaker_id": li["charID"],
                    "gender": ci.get("gender", "?"),
                    "credit_pos": ci.get("credit_pos", "?"),
                    "text": li["text"],
                })
            if len(turns) >= 2:
                movie_convos.append({
                    "participants": [
                        characters.get(convo["char1"], {}).get("name", "?"),
                        characters.get(convo["char2"], {}).get("name", "?"),
                    ],
                    "turns": turns,
                })
        
        # Parallel subtitle pairs for this movie
        parallel_pairs = parallel_data.get(imdb_id, [])
        
        # Build record
        record = {
            "cornell_id": movie_id,
            "imdb_id": imdb_id,
            "tconst": imdb_info["tconst"],
            "title": info["title"],
            "year": info["year"],
            "rating": info["rating"],
            "genres": info["genres"],
            "num_characters": len(movie_chars),
            "num_conversations": len(movie_convos),
            "num_parallel_pairs": len(parallel_pairs),
            "characters": {
                cid: {"name": ci["name"], "gender": ci["gender"],
                      "credit_pos": ci["credit_pos"]}
                for cid, ci in movie_chars.items()
            },
            "conversations": movie_convos,
            "parallel_subtitles": parallel_pairs,
        }
        
        out_path = os.path.join(OUTPUT_DIR, f"{movie_id}_{imdb_id}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        summary.append({
            "cornell_id": movie_id,
            "imdb_id": imdb_id,
            "title": info["title"],
            "year": info["year"],
            "genres": info["genres"],
            "characters": len(movie_chars),
            "conversations": len(movie_convos),
            "parallel_pairs": len(parallel_pairs),
        })
    
    # Sort by parallel pair count
    summary.sort(key=lambda x: x["parallel_pairs"], reverse=True)
    
    with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print summary
    with_parallel = sum(1 for s in summary if s["parallel_pairs"] > 0)
    total_parallel = sum(s["parallel_pairs"] for s in summary)
    total_convos = sum(s["conversations"] for s in summary)
    
    print(f"{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total movies:                    {len(summary)}")
    print(f"  Movies with EN↔ZH parallel data: {with_parallel}")
    print(f"  Total EN conversations (Cornell): {total_convos}")
    print(f"  Total EN↔ZH parallel pairs:       {total_parallel}")
    print(f"\n  Top 20 movies by parallel pairs:")
    for s in summary[:20]:
        print(f"    {s['title']} ({s['year']}) — "
              f"{s['parallel_pairs']} pairs, "
              f"{s['conversations']} convos, "
              f"{s['characters']} chars")
    print(f"\n  Output: {OUTPUT_DIR}/")
    print(f"  Summary: {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()