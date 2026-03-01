"""
Cornell + OPUS Pipeline v4 — FINAL WORKING VERSION
====================================================

Root cause of previous failures:
  1. opustools queries http://opus.nlpl.eu/opusapi/ which now returns 
     HTTP 308 → HTTPS. opustools can't follow it → all downloads fail.
  2. Stale empty cache from failed runs was being reloaded.
  3. Alignment XML references .xml.gz but zip contains .xml (no gzip).

Solution: Bypass opustools entirely. Download from object.pouta.csc.fi.
Auto-detect stale cache. Handle .xml/.xml.gz mismatch.

Downloads:
  - Alignment XML:  119 MB  (document structure + IMDB IDs)
  - zh_cn.zip:    1,593 MB  (Chinese subtitle XMLs)
  - en.zip:       SKIP      (24 GB — Cornell has the English text)

Requirements: Python stdlib only
"""

import os
import re
import ast
import csv
import gzip
import json
import zipfile
import urllib.request
from collections import defaultdict
from xml.etree import ElementTree as ET

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = "./data"
CORNELL_DIR = os.path.join(DATA_DIR, "cornell")
IMDB_DIR = os.path.join(DATA_DIR, "imdb")
OPUS_DIR = os.path.join(DATA_DIR, "opus")
OUTPUT_DIR = os.path.join(DATA_DIR, "combined")

CORNELL_URL = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
DELIMITER = " +++$+++ "

# Verified working URLs (from probe script)
OPUS_ALIGNMENT_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/en-zh_cn.xml.gz"
OPUS_ZH_ZIP_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/zh_cn.zip"


# =============================================================================
# STEP 1: Cornell
# =============================================================================

def download_cornell():
    os.makedirs(CORNELL_DIR, exist_ok=True)
    zip_path = os.path.join(CORNELL_DIR, "cornell.zip")
    extract_dir = os.path.join(CORNELL_DIR, "cornell movie-dialogs corpus")
    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            print("Downloading Cornell Movie-Dialogs Corpus (~10MB)...")
            urllib.request.urlretrieve(CORNELL_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(CORNELL_DIR)
    else:
        print(f"Cornell already at: {extract_dir}")
    return extract_dir


def parse_cornell(corpus_dir):
    enc = "iso-8859-1"
    movies = {}
    with open(os.path.join(corpus_dir, "movie_titles_metadata.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 6:
                mid = p[0].strip()
                movies[mid] = {
                    "title": p[1].strip(),
                    "year": re.sub(r'[^0-9]', '', p[2].strip())[:4],
                    "rating": p[3].strip(), "votes": p[4].strip(),
                    "genres": p[5].strip(),
                }
    print(f"  {len(movies)} movies")

    characters = {}
    with open(os.path.join(corpus_dir, "movie_characters_metadata.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 6:
                characters[p[0].strip()] = {
                    "name": p[1].strip(), "movieID": p[2].strip(),
                    "movie_title": p[3].strip(), "gender": p[4].strip(),
                    "credit_pos": p[5].strip(),
                }
    print(f"  {len(characters)} characters")

    lines = {}
    with open(os.path.join(corpus_dir, "movie_lines.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 5:
                lines[p[0].strip()] = {
                    "charID": p[1].strip(), "movieID": p[2].strip(),
                    "char_name": p[3].strip(),
                    "text": p[4].strip() if p[4].strip() else "",
                }
    print(f"  {len(lines)} lines")

    conversations = []
    with open(os.path.join(corpus_dir, "movie_conversations.txt"), 'r', encoding=enc) as f:
        for line in f:
            p = line.strip().split(DELIMITER)
            if len(p) >= 4:
                try:
                    line_ids = ast.literal_eval(p[3].strip())
                except:
                    line_ids = re.findall(r"'(L\d+)'", p[3])
                conversations.append({
                    "char1": p[0].strip(), "char2": p[1].strip(),
                    "movieID": p[2].strip(), "line_ids": line_ids,
                })
    print(f"  {len(conversations)} conversations")
    return movies, characters, lines, conversations


# =============================================================================
# STEP 2: IMDB resolution
# =============================================================================

def download_imdb_basics():
    os.makedirs(IMDB_DIR, exist_ok=True)
    gz_path = os.path.join(IMDB_DIR, "title.basics.tsv.gz")
    if not os.path.exists(gz_path):
        print("Downloading IMDb title.basics.tsv.gz (~150MB)...")
        urllib.request.urlretrieve(IMDB_BASICS_URL, gz_path)
    else:
        print(f"IMDb basics already at: {gz_path}")
    return gz_path


def build_imdb_lookup(gz_path):
    print("Building IMDb title lookup (~30 seconds)...")
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
    print(f"  {len(lookup)} entries")
    return lookup


def resolve_imdb_ids(movies, imdb_lookup):
    cache_path = os.path.join(CORNELL_DIR, "imdb_mapping.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    mapping = {}
    for mid, info in movies.items():
        title = info["title"].lower()
        year = info["year"]
        tc = imdb_lookup.get((title, year))
        if not tc and title.startswith("the "):
            tc = imdb_lookup.get((title[4:], year))
        if not tc:
            tc = imdb_lookup.get(("the " + title, year))
        if not tc and year.isdigit():
            for off in [-1, 1]:
                tc = imdb_lookup.get((title, str(int(year) + off)))
                if tc:
                    break
        if tc:
            mapping[mid] = {"imdb_id": tc.replace("tt", ""), "tconst": tc,
                            "title": info["title"], "year": year}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Matched {len(mapping)}/{len(movies)} movies")
    return mapping


# =============================================================================
# STEP 3: OPUS — direct download, bypass opustools
# =============================================================================

def download_with_progress(url, dest_path):
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  Already exists: {dest_path} ({size_mb:.1f} MB)")
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
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB ({pct:.0f}%)",
                              end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def parse_alignment_for_movies(alignment_gz_path, target_imdb_ids):
    """
    Parse XCES alignment XML for document pairs matching our movies.
    
    Path format: lang/YEAR/IMDBID/UPLOAD.xml.gz
      e.g. en/1999/147800/4681542.xml.gz
    
    IMDB IDs in paths have NO leading zeros.
    Cornell IDs have leading zeros (e.g. '0147800').
    We strip zeros from both sides for matching.
    """
    print("Parsing alignment XML for matching IMDB IDs...")
    
    # Build lookup: stripped_id -> original_id
    target_lookup = {}
    for tid in target_imdb_ids:
        stripped = str(tid).lstrip('0')
        target_lookup[stripped] = str(tid)
    
    print(f"  Looking for {len(target_lookup)} target movies")
    
    matched = defaultdict(list)
    total_linkgrps = 0
    
    with gzip.open(alignment_gz_path, 'rt', encoding='utf-8') as f:
        for event, elem in ET.iterparse(f, events=['start', 'end']):
            if event == 'start' and elem.tag == 'linkGrp':
                total_linkgrps += 1
                to_doc = elem.get('toDoc', '')
                from_doc = elem.get('fromDoc', '')
                
                # Determine which is EN and which is ZH
                if to_doc.startswith('en'):
                    en_doc, zh_doc = to_doc, from_doc
                else:
                    en_doc, zh_doc = from_doc, to_doc
                
                # Extract IMDB ID from position 2 in path
                # Format: lang/YEAR/IMDBID/UPLOAD.xml.gz
                found_imdb = None
                for doc in [en_doc, zh_doc]:
                    parts = doc.split('/')
                    if len(parts) >= 4 and parts[2].isdigit():
                        stripped = parts[2].lstrip('0')
                        if stripped in target_lookup:
                            found_imdb = target_lookup[stripped]
                            break
                
                if found_imdb:
                    # Count links in this linkGrp
                    # We'll count them in the 'end' event
                    elem.set('_matched_imdb', found_imdb)
                    elem.set('_en_doc', en_doc)
                    elem.set('_zh_doc', zh_doc)
                    elem.set('_link_count', '0')
                
            elif event == 'start' and elem.tag == 'link':
                # Find parent linkGrp
                pass
                
            elif event == 'end' and elem.tag == 'linkGrp':
                imdb = elem.get('_matched_imdb')
                if imdb:
                    # Count link children
                    link_count = len(list(elem.iter('link')))
                    matched[imdb].append({
                        'en_doc': elem.get('_en_doc'),
                        'zh_doc': elem.get('_zh_doc'),
                        'num_links': link_count,
                    })
                elem.clear()
                
                if total_linkgrps % 10000 == 0:
                    print(f"\r  Scanned {total_linkgrps} doc pairs, "
                          f"matched {len(matched)} movies...", end="", flush=True)
    
    print(f"\r  Scanned {total_linkgrps} document pairs total                    ")
    total_docs = sum(len(v) for v in matched.values())
    print(f"  Matched {len(matched)} movies ({total_docs} document pairs)")
    
    return dict(matched)


def extract_zh_subtitles(zh_zip_path, matched_movies):
    """
    Extract Chinese subtitle text from zh_cn.zip for matched movies.
    
    IMPORTANT: The alignment XML references paths like:
      zh_cn/2003/305357/5959539.xml.gz
    But the zip contains files WITHOUT .gz:
      OpenSubtitles/xml/zh_cn/2003/305357/5959539.xml
    
    We handle both cases.
    """
    print(f"Extracting Chinese subtitles from zip...")
    
    # Collect all zh_doc paths we need
    docs_needed = {}  # zh_doc_path -> imdb_id
    for imdb_id, doc_list in matched_movies.items():
        for doc_info in doc_list:
            docs_needed[doc_info['zh_doc']] = imdb_id
    
    print(f"  Need {len(docs_needed)} Chinese subtitle documents")
    
    result = defaultdict(dict)  # imdb_id -> {doc_path: {sent_id: text}}
    
    with zipfile.ZipFile(zh_zip_path, 'r') as zf:
        namelist = set(zf.namelist())
        
        # Show sample paths from zip for debugging
        sample = [n for n in list(namelist)[:100] if n.endswith('.xml') or n.endswith('.xml.gz')]
        if sample:
            print(f"  Zip sample path: {sample[0]}")
        
        found = 0
        not_found = 0
        
        for zh_doc, imdb_id in docs_needed.items():
            # Build candidate paths to search for in the zip
            # zh_doc from alignment: "zh_cn/2003/305357/5959539.xml.gz"
            # Actual zip path:       "OpenSubtitles/xml/zh_cn/2003/305357/5959539.xml"
            
            # Strip .gz if present (alignment says .xml.gz, zip has .xml)
            zh_doc_no_gz = zh_doc[:-3] if zh_doc.endswith('.gz') else zh_doc
            zh_doc_with_gz = zh_doc if zh_doc.endswith('.gz') else zh_doc + '.gz'
            
            candidates = [
                # With OpenSubtitles/xml/ prefix (most likely based on diagnostic)
                f"OpenSubtitles/xml/{zh_doc_no_gz}",
                f"OpenSubtitles/xml/{zh_doc_with_gz}",
                # Without prefix
                zh_doc_no_gz,
                zh_doc_with_gz,
                # With OpenSubtitles/ prefix only
                f"OpenSubtitles/{zh_doc_no_gz}",
                f"OpenSubtitles/{zh_doc_with_gz}",
            ]
            
            zip_path = None
            for c in candidates:
                if c in namelist:
                    zip_path = c
                    break
            
            if not zip_path:
                not_found += 1
                continue
            
            found += 1
            
            try:
                with zf.open(zip_path) as inner:
                    raw = inner.read()
                    # Decompress if the file inside zip is itself gzipped
                    try:
                        content = gzip.decompress(raw)
                    except (gzip.BadGzipFile, OSError):
                        content = raw  # Not gzipped, use as-is
                    
                    sentences = {}
                    root = ET.fromstring(content)
                    for s_elem in root.iter('s'):
                        sid = s_elem.get('id', '')
                        words = [w.text for w in s_elem.iter('w') if w.text]
                        if words:
                            sentences[sid] = ''.join(words)
                        else:
                            text = ''.join(s_elem.itertext()).strip()
                            if text:
                                sentences[sid] = text
                    
                    if sentences:
                        result[imdb_id][zh_doc] = sentences
                        
            except Exception:
                pass
            
            if found % 200 == 0:
                print(f"\r  Extracted {found}/{len(docs_needed)} "
                      f"({not_found} not found)...", end="", flush=True)
    
    print(f"\r  Done: {found} extracted, {not_found} not found in zip          ")
    
    # For each movie, pick the subtitle upload with the most sentences
    zh_lines = {}
    for imdb_id, doc_dict in result.items():
        best_doc = max(doc_dict.values(), key=len)
        sorted_sents = []
        for sid in sorted(best_doc.keys(),
                          key=lambda x: int(re.sub(r'[^0-9]', '', x) or 0)):
            sorted_sents.append(best_doc[sid])
        zh_lines[imdb_id] = sorted_sents
    
    return zh_lines


def download_opus_en_zh(target_imdb_ids):
    """Main OPUS function."""
    os.makedirs(OPUS_DIR, exist_ok=True)
    
    # Check cache — but invalidate if empty (stale from failed runs)
    cache_path = os.path.join(OPUS_DIR, "opus_matched.json")
    if os.path.exists(cache_path):
        file_size = os.path.getsize(cache_path)
        if file_size > 10:  # Non-trivial cache (empty dict = 2 bytes)
            with open(cache_path, 'r') as f:
                data = json.load(f)
            if data:  # Actually has content
                total = sum(len(v) for v in data.values())
                print(f"Loaded cached OPUS data: {len(data)} movies, {total} ZH lines")
                return data
        # Stale empty cache — delete it
        print(f"Removing stale empty cache: {cache_path}")
        os.remove(cache_path)
    
    alignment_path = os.path.join(OPUS_DIR, "en-zh_cn.xml.gz")
    zh_zip_path = os.path.join(OPUS_DIR, "zh_cn.zip")
    
    # Step 1: Download alignment XML (119 MB)
    print("\n[3a] Downloading alignment file (119 MB)...")
    if not download_with_progress(OPUS_ALIGNMENT_URL, alignment_path):
        return {}
    
    # Step 2: Parse alignment to find matching movies
    print("\n[3b] Parsing alignment for target movies...")
    matched_movies = parse_alignment_for_movies(alignment_path, target_imdb_ids)
    
    if not matched_movies:
        print("No matching movies found in alignment!")
        return {}
    
    # Show coverage
    print(f"\nFound {len(matched_movies)} Cornell movies with ZH subtitles!")
    for imdb_id in sorted(matched_movies.keys(),
                          key=lambda x: -sum(d['num_links'] for d in matched_movies[x]))[:15]:
        docs = matched_movies[imdb_id]
        total_links = sum(d['num_links'] for d in docs)
        print(f"  IMDB {imdb_id}: {len(docs)} uploads, ~{total_links} lines")
    
    # Step 3: Download zh_cn.zip (1.6 GB)
    print(f"\n[3c] Downloading Chinese subtitles (~1.6 GB, one-time)...")
    if not download_with_progress(OPUS_ZH_ZIP_URL, zh_zip_path):
        print(f"Manual download: {OPUS_ZH_ZIP_URL}")
        print(f"Save to: {os.path.abspath(zh_zip_path)}")
        return {}
    
    # Step 4: Extract Chinese subtitles for matched movies
    print(f"\n[3d] Extracting ZH subtitles for {len(matched_movies)} movies...")
    result = extract_zh_subtitles(zh_zip_path, matched_movies)
    
    # Cache results
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    
    total_lines = sum(len(v) for v in result.values())
    print(f"\nOPUS complete: {len(result)} movies, {total_lines} ZH subtitle lines")
    return result


# =============================================================================
# STEP 4: Build combined dataset
# =============================================================================

def build_dataset(movies, characters, lines, conversations,
                  imdb_mapping, opus_data=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    for movie_id, info in movies.items():
        imdb_info = imdb_mapping.get(movie_id)
        if not imdb_info:
            continue

        imdb_id = imdb_info["imdb_id"]
        movie_chars = {cid: ci for cid, ci in characters.items()
                       if ci["movieID"] == movie_id}

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
                    "line_id": lid, "speaker": li["char_name"],
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

        zh_lines = opus_data.get(imdb_id, []) if opus_data else []

        record = {
            "cornell_id": movie_id, "imdb_id": imdb_id,
            "tconst": imdb_info["tconst"],
            "title": info["title"], "year": info["year"],
            "rating": info["rating"], "genres": info["genres"],
            "num_characters": len(movie_chars),
            "num_conversations": len(movie_convos),
            "num_zh_subtitle_lines": len(zh_lines),
            "characters": {cid: {"name": ci["name"], "gender": ci["gender"],
                                  "credit_pos": ci["credit_pos"]}
                           for cid, ci in movie_chars.items()},
            "conversations": movie_convos,
            "zh_subtitles": zh_lines,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{movie_id}_{imdb_id}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        summary.append({
            "cornell_id": movie_id, "imdb_id": imdb_id,
            "title": info["title"], "year": info["year"],
            "genres": info["genres"],
            "conversations": len(movie_convos),
            "zh_subtitle_lines": len(zh_lines),
            "characters": len(movie_chars),
        })

    summary.sort(key=lambda x: x["zh_subtitle_lines"], reverse=True)

    with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with_zh = sum(1 for s in summary if s["zh_subtitle_lines"] > 0)
    total_zh = sum(s["zh_subtitle_lines"] for s in summary)
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total movies with IMDB IDs:        {len(summary)}")
    print(f"Movies with ZH subtitles:          {with_zh}")
    print(f"Total EN conversations (Cornell):  {sum(s['conversations'] for s in summary)}")
    print(f"Total ZH subtitle lines (OPUS):    {total_zh}")
    print(f"\nTop 20 movies by ZH subtitle coverage:")
    for s in summary[:20]:
        print(f"  {s['title']} ({s['year']}) — "
              f"{s['zh_subtitle_lines']} ZH, {s['conversations']} EN convos")
    print(f"\nOutput: {OUTPUT_DIR}/")
    return summary


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(skip_opus=False):
    print("=" * 60)
    print("STEP 1: Cornell Movie-Dialogs Corpus")
    print("=" * 60)
    corpus_dir = download_cornell()
    movies, characters, lines, conversations = parse_cornell(corpus_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Resolve IMDB IDs")
    print("=" * 60)
    cache_path = os.path.join(CORNELL_DIR, "imdb_mapping.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            imdb_mapping = json.load(f)
        print(f"Loaded cached mapping ({len(imdb_mapping)} movies)")
    else:
        gz_path = download_imdb_basics()
        imdb_lookup = build_imdb_lookup(gz_path)
        imdb_mapping = resolve_imdb_ids(movies, imdb_lookup)

    opus_data = None
    if not skip_opus:
        print("\n" + "=" * 60)
        print("STEP 3: OPUS EN-ZH Subtitles (direct download)")
        print("=" * 60)
        target_ids = set(v["imdb_id"] for v in imdb_mapping.values())
        opus_data = download_opus_en_zh(target_ids)

    print("\n" + "=" * 60)
    print("STEP 4: Build combined dataset")
    print("=" * 60)
    summary = build_dataset(movies, characters, lines, conversations,
                            imdb_mapping, opus_data)
    return summary


if __name__ == "__main__":
    run_pipeline(skip_opus=False)