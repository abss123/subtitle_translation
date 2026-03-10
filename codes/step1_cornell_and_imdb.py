"""
Step 1: Cornell Movie-Dialogs + IMDB ID Resolution
====================================================
Downloads Cornell corpus (10 MB) and IMDb basics (150 MB).
Parses screenplay dialogue with speaker labels.
Resolves movie titles to IMDB IDs for linking with OPUS subtitles.

Produces:
  data/cornell/cornell movie-dialogs corpus/  (raw corpus files)
  data/cornell/imdb_mapping.json              (cornell_id → imdb_id)
  data/cornell/parsed_cornell.json            (all parsed data)

Run: python step1_cornell_and_imdb.py
"""

import os
import re
import ast
import csv
import gzip
import json
import zipfile
import urllib.request

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = "./data"
CORNELL_DIR = os.path.join(DATA_DIR, "cornell")
IMDB_DIR = os.path.join(DATA_DIR, "imdb")

CORNELL_URL = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
DELIMITER = " +++$+++ "


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_with_progress(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return True
    print(f"  Downloading: {url}")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        print(f"\r  {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB "
                              f"({downloaded/total*100:.0f}%)", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# =============================================================================
# CORNELL PARSING
# =============================================================================

def download_and_extract_cornell():
    os.makedirs(CORNELL_DIR, exist_ok=True)
    zip_path = os.path.join(CORNELL_DIR, "cornell.zip")
    extract_dir = os.path.join(CORNELL_DIR, "cornell movie-dialogs corpus")
    if not os.path.exists(extract_dir):
        download_with_progress(CORNELL_URL, zip_path)
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(CORNELL_DIR)
    else:
        print(f"  Already extracted: {extract_dir}")
    return extract_dir


def parse_cornell(corpus_dir):
    enc = "iso-8859-1"

    # Movies
    movies = {}
    with open(os.path.join(corpus_dir, "movie_titles_metadata.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 6:
                mid = p[0].strip()
                movies[mid] = {
                    "title": p[1].strip(),
                    "year": re.sub(r'[^0-9]', '', p[2].strip())[:4],
                    "rating": p[3].strip(),
                    "votes": p[4].strip(),
                    "genres": p[5].strip(),
                }
    print(f"  {len(movies)} movies")

    # Characters
    characters = {}
    with open(os.path.join(corpus_dir, "movie_characters_metadata.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 6:
                characters[p[0].strip()] = {
                    "name": p[1].strip(),
                    "movieID": p[2].strip(),
                    "movie_title": p[3].strip(),
                    "gender": p[4].strip(),
                    "credit_pos": p[5].strip(),
                }
    print(f"  {len(characters)} characters")

    # Lines
    lines = {}
    with open(os.path.join(corpus_dir, "movie_lines.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 5:
                lines[p[0].strip()] = {
                    "charID": p[1].strip(),
                    "movieID": p[2].strip(),
                    "char_name": p[3].strip(),
                    "text": p[4].strip() if p[4].strip() else "",
                }
    print(f"  {len(lines)} dialogue lines")

    # Conversations
    conversations = []
    with open(os.path.join(corpus_dir, "movie_conversations.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 4:
                try:
                    line_ids = ast.literal_eval(p[3].strip())
                except Exception:
                    line_ids = re.findall(r"'(L\d+)'", p[3])
                conversations.append({
                    "char1": p[0].strip(),
                    "char2": p[1].strip(),
                    "movieID": p[2].strip(),
                    "line_ids": line_ids,
                })
    print(f"  {len(conversations)} conversations")

    return movies, characters, lines, conversations


# =============================================================================
# IMDB RESOLUTION
# =============================================================================

def resolve_imdb_ids(movies):
    """
    Match Cornell movie titles to IMDB IDs using the free IMDb basics dataset.
    Downloads ~150 MB TSV, builds a (title, year) → tconst lookup, then
    matches each Cornell movie. Results are cached to imdb_mapping.json.
    """
    cache_path = os.path.join(CORNELL_DIR, "imdb_mapping.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            mapping = json.load(f)
        print(f"  Loaded cached IMDB mapping: {len(mapping)} movies")
        return mapping

    # Download IMDb basics
    os.makedirs(IMDB_DIR, exist_ok=True)
    gz_path = os.path.join(IMDB_DIR, "title.basics.tsv.gz")
    download_with_progress(IMDB_BASICS_URL, gz_path)

    # Build lookup
    print("  Building title lookup (~30 seconds)...")
    lookup = {}
    with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get("titleType") not in ("movie", "tvMovie"):
                continue
            title = row.get("primaryTitle", "").strip().lower()
            orig = row.get("originalTitle", "").strip().lower()
            year = row.get("startYear", "").strip()
            tc = row.get("tconst", "").strip()
            if title and year and tc:
                lookup[(title, year)] = tc
            if orig and year and tc and orig != title:
                lookup[(orig, year)] = tc
    print(f"  {len(lookup)} IMDb entries loaded")

    # Match
    mapping = {}
    for mid, info in movies.items():
        title = info["title"].lower()
        year = info["year"]
        tc = lookup.get((title, year))
        if not tc and title.startswith("the "):
            tc = lookup.get((title[4:], year))
        if not tc:
            tc = lookup.get(("the " + title, year))
        if not tc and year.isdigit():
            for off in [-1, 1]:
                tc = lookup.get((title, str(int(year) + off)))
                if tc:
                    break
        if tc:
            mapping[mid] = {
                "imdb_id": tc.replace("tt", ""),
                "tconst": tc,
                "title": info["title"],
                "year": year,
            }

    with open(cache_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"  Matched {len(mapping)}/{len(movies)} movies → saved to {cache_path}")
    return mapping


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 1a: Download Cornell Movie-Dialogs Corpus")
    print("=" * 60)
    corpus_dir = download_and_extract_cornell()

    print("\n" + "=" * 60)
    print("STEP 1b: Parse Cornell data")
    print("=" * 60)
    movies, characters, lines, conversations = parse_cornell(corpus_dir)

    # Save parsed data for use by later scripts
    parsed = {
        "movies": movies,
        "characters": characters,
        "lines": lines,
        "conversations": conversations,
    }
    parsed_path = os.path.join(CORNELL_DIR, "parsed_cornell.json")
    with open(parsed_path, 'w', encoding='utf-8') as f:
        json.dump(parsed, f, ensure_ascii=False)
    print(f"  Saved parsed data to {parsed_path}")

    print("\n" + "=" * 60)
    print("STEP 1c: Resolve IMDB IDs")
    print("=" * 60)
    imdb_mapping = resolve_imdb_ids(movies)

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    print(f"  Movies:        {len(movies)}")
    print(f"  Characters:    {len(characters)}")
    print(f"  Lines:         {len(lines)}")
    print(f"  Conversations: {len(conversations)}")
    print(f"  IMDB matched:  {len(imdb_mapping)}")
    print(f"\n  Files created:")
    print(f"    {parsed_path}")
    print(f"    {os.path.join(CORNELL_DIR, 'imdb_mapping.json')}")


if __name__ == "__main__":
    main()