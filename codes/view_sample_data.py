"""
View sample data from the Cornell + OPUS combined dataset.
Run after cornell_opus_pipeline_v4.py completes.
"""

import os
import json
import glob

OUTPUT_DIR = "../data/combined"

# ─────────────────────────────────────────────────────────
# 1. Load summary
# ─────────────────────────────────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, "summary.json")
if not os.path.exists(summary_path):
    print(f"ERROR: {summary_path} not found. Run the pipeline first.")
    exit(1)

with open(summary_path, 'r') as f:
    summary = json.load(f)

with_zh = [s for s in summary if s["zh_subtitle_lines"] > 0]
without_zh = [s for s in summary if s["zh_subtitle_lines"] == 0]

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Total movies:              {len(summary)}")
print(f"Movies WITH ZH subtitles:  {len(with_zh)}")
print(f"Movies without ZH:         {len(without_zh)}")
print(f"Total EN conversations:    {sum(s['conversations'] for s in summary)}")
print(f"Total ZH subtitle lines:   {sum(s['zh_subtitle_lines'] for s in summary)}")

# ─────────────────────────────────────────────────────────
# 2. Top movies by ZH coverage
# ─────────────────────────────────────────────────────────
if with_zh:
    print(f"\n{'=' * 70}")
    print("TOP 25 MOVIES BY ZH SUBTITLE LINES")
    print(f"{'=' * 70}")
    for i, s in enumerate(with_zh[:25], 1):
        print(f"  {i:2d}. {s['title']} ({s['year']}) — "
              f"{s['zh_subtitle_lines']} ZH lines, "
              f"{s['conversations']} convos, {s['characters']} chars")

# ─────────────────────────────────────────────────────────
# 3. Detailed view of best movie
# ─────────────────────────────────────────────────────────
def view_movie(movie_entry):
    """Load and display a single movie's full data."""
    pattern = os.path.join(OUTPUT_DIR, f"{movie_entry['cornell_id']}_{movie_entry['imdb_id']}.json")
    files = glob.glob(pattern)
    if not files:
        print(f"  File not found: {pattern}")
        return
    
    with open(files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'─' * 70}")
    print(f"MOVIE: {data['title']} ({data['year']})")
    print(f"{'─' * 70}")
    print(f"  Cornell ID:  {data['cornell_id']}")
    print(f"  IMDB ID:     tt{data['imdb_id']}")
    print(f"  Rating:      {data['rating']}")
    print(f"  Genres:      {data['genres']}")
    print(f"  Characters:  {data['num_characters']}")
    print(f"  EN Convos:   {data['num_conversations']}")
    print(f"  ZH Lines:    {data['num_zh_subtitle_lines']}")
    
    # Characters
    print(f"\n  CHARACTERS:")
    for cid, ci in list(data['characters'].items())[:10]:
        print(f"    {ci['name']:20s}  gender={ci['gender']:4s}  credit={ci['credit_pos']}")
    if len(data['characters']) > 10:
        print(f"    ... and {len(data['characters']) - 10} more")
    
    # Sample conversations (first 3)
    print(f"\n  SAMPLE CONVERSATIONS (EN):")
    for i, convo in enumerate(data['conversations'][:3]):
        print(f"\n  --- Conversation {i+1} ({convo['participants'][0]} ↔ {convo['participants'][1]}) ---")
        for turn in convo['turns'][:6]:
            speaker = turn['speaker']
            text = turn['text'][:100] + ('...' if len(turn['text']) > 100 else '')
            print(f"    {speaker:15s}: {text}")
        if len(convo['turns']) > 6:
            print(f"    ... ({len(convo['turns'])} turns total)")
    
    # Sample ZH subtitles
    if data['zh_subtitles']:
        print(f"\n  SAMPLE ZH SUBTITLES (first 20 lines):")
        for i, line in enumerate(data['zh_subtitles'][:20]):
            print(f"    {i+1:4d}: {line}")
        if len(data['zh_subtitles']) > 20:
            print(f"    ... ({len(data['zh_subtitles'])} lines total)")
        
        print(f"\n  SAMPLE ZH SUBTITLES (middle section):")
        mid = len(data['zh_subtitles']) // 2
        for i, line in enumerate(data['zh_subtitles'][mid:mid+10]):
            print(f"    {mid+i+1:4d}: {line}")
    else:
        print(f"\n  (No ZH subtitles for this movie)")
    
    return data


# Show detailed view for top 3 movies with ZH
if with_zh:
    print(f"\n{'=' * 70}")
    print("DETAILED VIEW — TOP 3 MOVIES WITH ZH SUBTITLES")
    print(f"{'=' * 70}")
    for entry in with_zh[:3]:
        view_movie(entry)

else:
    # If no ZH movies, show a sample movie anyway
    print(f"\n{'=' * 70}")
    print("NO ZH SUBTITLES FOUND — showing sample EN-only movie")
    print(f"{'=' * 70}")
    view_movie(summary[0])

# ─────────────────────────────────────────────────────────
# 4. Quick stats on ZH coverage
# ─────────────────────────────────────────────────────────
if with_zh:
    zh_counts = [s['zh_subtitle_lines'] for s in with_zh]
    print(f"\n{'=' * 70}")
    print("ZH COVERAGE STATISTICS")
    print(f"{'=' * 70}")
    print(f"  Movies with ZH:     {len(zh_counts)}")
    print(f"  Min ZH lines:       {min(zh_counts)}")
    print(f"  Max ZH lines:       {max(zh_counts)}")
    print(f"  Mean ZH lines:      {sum(zh_counts) / len(zh_counts):.0f}")
    print(f"  Median ZH lines:    {sorted(zh_counts)[len(zh_counts)//2]}")
    print(f"  Total ZH lines:     {sum(zh_counts)}")
    
    # Genre breakdown
    genre_zh = {}
    for s in with_zh:
        genres = s.get('genres', '[]')
        try:
            gs = eval(genres) if genres.startswith('[') else [genres]
        except:
            gs = [genres]
        for g in gs:
            g = g.strip().strip("'\"")
            if g:
                genre_zh[g] = genre_zh.get(g, 0) + 1
    
    if genre_zh:
        print(f"\n  ZH coverage by genre:")
        for genre, count in sorted(genre_zh.items(), key=lambda x: -x[1])[:15]:
            print(f"    {genre:20s}: {count} movies")