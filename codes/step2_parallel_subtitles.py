"""
Step 2: Download Aligned EN↔ZH Subtitle Pairs from OPUS
=========================================================
Requires: Step 1 completed (needs data/cornell/imdb_mapping.json)

Downloads:
  - Alignment XML (119 MB) — tells us which EN↔ZH subtitle docs are paired
  - zh_cn.zip (1.6 GB) — Chinese subtitle XML files
  - EN subtitle files selectively from en.zip via HTTP range requests
    (only the files we need, NOT the full 24 GB)

How alignment works:
  The alignment XML contains <linkGrp> elements for each document pair:
    <linkGrp toDoc="en/1994/110912/4567.xml.gz" 
             fromDoc="zh_cn/1994/110912/8901.xml.gz">
      <link xtargets="s1;s1" />     ← EN sentence 1 maps to ZH sentence 1
      <link xtargets="s2 s3;s2" />  ← EN sentences 2+3 map to ZH sentence 2
    </linkGrp>

  Each <link> has an xtargets attribute with explicit sentence ID mappings.
  We use these to pair EN and ZH text exactly — no offset guessing.

Produces:
  data/opus/parallel_en_zh.json — {imdb_id: [{en: "...", zh: "..."}, ...]}

Run: python step2_parallel_subtitles.py
"""

import os
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
OPUS_DIR = os.path.join(DATA_DIR, "opus")
CORNELL_DIR = os.path.join(DATA_DIR, "cornell")
PARALLEL_CACHE = os.path.join(OPUS_DIR, "parallel_en_zh.json")

OPUS_ALIGNMENT_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/en-zh_cn.xml.gz"
OPUS_ZH_ZIP_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/zh_cn.zip"
OPUS_EN_ZIP_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/en.zip"


# =============================================================================
# HTTP RANGE-BACKED ZIP READER
# =============================================================================

class HTTPRangeFile:
    """
    File-like object backed by HTTP range requests.
    Allows zipfile.ZipFile to read specific entries from a remote zip
    without downloading the entire file.
    
    How it works:
    - zipfile first reads the central directory (last ~1-5 MB of the zip)
    - Then for each file we want, it reads just that file's byte range
    - Total download for ~400 movies: a few hundred MB instead of 24 GB
    """
    def __init__(self, url):
        self.url = url
        self._pos = 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as resp:
            self._size = int(resp.headers['Content-Length'])
        print(f"  Remote zip: {self._size / 1024/1024:.0f} MB "
              f"(only fetching files we need)")

    def seekable(self): return True
    def readable(self): return True

    def seek(self, offset, whence=0):
        if whence == 0:   self._pos = offset
        elif whence == 1: self._pos += offset
        elif whence == 2: self._pos = self._size + offset
        return self._pos

    def tell(self): return self._pos

    def read(self, size=-1):
        if size == 0: return b''
        if size < 0: size = self._size - self._pos
        if self._pos >= self._size: return b''
        end = min(self._pos + size - 1, self._size - 1)
        req = urllib.request.Request(self.url)
        req.add_header('Range', f'bytes={self._pos}-{end}')
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except Exception as e:
            print(f"\n  Range request failed: {e}")
            return b''
        self._pos += len(data)
        return data


# =============================================================================
# HELPERS
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
                    if not chunk: break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        print(f"\r  {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB "
                              f"({downloaded/total*100:.0f}%)", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        if os.path.exists(dest_path): os.remove(dest_path)
        return False


def resolve_zip_path(doc_path, namelist):
    """
    Find actual path inside a zip for a given alignment doc path.
    Alignment says: zh_cn/2003/305357/5959539.xml.gz
    Zip contains:  OpenSubtitles/xml/zh_cn/2003/305357/5959539.xml (no .gz!)
    """
    no_gz = doc_path[:-3] if doc_path.endswith('.gz') else doc_path
    with_gz = doc_path if doc_path.endswith('.gz') else doc_path + '.gz'
    for candidate in [f"OpenSubtitles/xml/{no_gz}", f"OpenSubtitles/xml/{with_gz}",
                      no_gz, with_gz]:
        if candidate in namelist:
            return candidate
    return None


def extract_sentences_from_xml(content, join_char=' '):
    """Parse OPUS subtitle XML → {sentence_id: text}."""
    sentences = {}
    try:
        root = ET.fromstring(content)
        for s in root.iter('s'):
            sid = s.get('id', '')
            words = [w.text for w in s.iter('w') if w.text]
            if words:
                sentences[sid] = join_char.join(words)
            else:
                text = ''.join(s.itertext()).strip()
                if text:
                    sentences[sid] = text
    except ET.ParseError:
        pass
    return sentences


def read_xml_from_zip(zf, zip_path, join_char=' '):
    """Read an XML subtitle file from inside a zip, handling gzip."""
    raw = zf.read(zip_path)
    try:
        content = gzip.decompress(raw)
    except (gzip.BadGzipFile, OSError):
        content = raw
    return extract_sentences_from_xml(content, join_char)


# =============================================================================
# STEP 2a: PARSE ALIGNMENT XML
# =============================================================================

def parse_alignment_xml(alignment_gz_path, target_imdb_ids):
    """
    Parse the XCES alignment XML and extract xtargets for matched movies.
    
    The alignment XML structure:
      <cesAlign>
        <linkGrp toDoc="en/YEAR/IMDBID/UPLOAD.xml.gz" 
                 fromDoc="zh_cn/YEAR/IMDBID/UPLOAD.xml.gz">
          <link xtargets="s1;s1" />
          <link xtargets="s2 s3;s2" />
        </linkGrp>
        ...
      </cesAlign>
    
    IMDB IDs in paths have NO leading zeros (e.g., "110912")
    Cornell IMDB IDs have leading zeros (e.g., "0110912")
    We strip zeros from both sides for matching.
    
    Returns: {imdb_id: [{en_doc, zh_doc, alignments: [(en_sids, zh_sids)]}]}
    """
    print("  Parsing alignment XML for matching movies...")
    
    # Build lookup: stripped_id → original zero-padded id
    target_lookup = {str(tid).lstrip('0'): str(tid) for tid in target_imdb_ids}
    
    matched = defaultdict(list)
    total = 0
    
    with gzip.open(alignment_gz_path, 'rt', encoding='utf-8') as f:
        for event, elem in ET.iterparse(f, events=['end']):
            if elem.tag != 'linkGrp':
                continue
            total += 1
            
            to_doc = elem.get('toDoc', '')
            from_doc = elem.get('fromDoc', '')
            
            # Figure out which is EN and which is ZH
            en_doc = to_doc if to_doc.startswith('en') else from_doc
            zh_doc = from_doc if from_doc.startswith('zh') else to_doc
            
            # Extract IMDB ID from path position 2: lang/YEAR/IMDBID/UPLOAD.xml.gz
            found_imdb = None
            for doc in [en_doc, zh_doc]:
                parts = doc.split('/')
                if len(parts) >= 4 and parts[2].isdigit():
                    stripped = parts[2].lstrip('0')
                    if stripped in target_lookup:
                        found_imdb = target_lookup[stripped]
                        break
            
            if found_imdb:
                # Extract sentence-level alignment from xtargets
                alignments = []
                for link in elem.iter('link'):
                    xt = link.get('xtargets', '')
                    if ';' not in xt:
                        continue
                    en_part, zh_part = xt.split(';')
                    en_sids = en_part.strip().split() if en_part.strip() else []
                    zh_sids = zh_part.strip().split() if zh_part.strip() else []
                    # Only keep 1:1 or few:few alignments where both sides have text
                    if en_sids and zh_sids:
                        alignments.append((en_sids, zh_sids))
                
                if alignments:
                    matched[found_imdb].append({
                        'en_doc': en_doc,
                        'zh_doc': zh_doc,
                        'alignments': alignments,
                    })
            
            elem.clear()
            if total % 10000 == 0:
                print(f"\r  Scanned {total} doc pairs, "
                      f"matched {len(matched)} movies...", end="", flush=True)
    
    print(f"\r  Scanned {total} document pairs, "
          f"matched {len(matched)} movies              ")
    return dict(matched)


# =============================================================================
# STEP 2b: EXTRACT PARALLEL PAIRS
# =============================================================================

def extract_parallel_pairs(matched_movies, zh_zip_path, en_zip_url):
    """
    For each matched movie:
      1. Pick the best subtitle pair (most aligned sentences)
      2. Extract ZH sentences from local zh_cn.zip
      3. Extract EN sentences from remote en.zip via HTTP range requests
      4. Join using xtargets for exact EN↔ZH alignment
    """
    
    # Pick best doc pair per movie (most alignments = best subtitle upload)
    best_docs = {}
    for imdb_id, doc_list in matched_movies.items():
        best = max(doc_list, key=lambda d: len(d['alignments']))
        best_docs[imdb_id] = best
    
    en_docs_needed = {d['en_doc'] for d in best_docs.values()}
    zh_docs_needed = {d['zh_doc'] for d in best_docs.values()}
    print(f"  Need {len(en_docs_needed)} EN + {len(zh_docs_needed)} ZH subtitle files")
    
    # ── ZH: extract from local zip ──
    print(f"\n  Extracting ZH sentences from local zh_cn.zip...")
    zh_sentences = {}  # zh_doc_path → {sentence_id: text}
    with zipfile.ZipFile(zh_zip_path, 'r') as zf:
        namelist = set(zf.namelist())
        for zh_doc in zh_docs_needed:
            zpath = resolve_zip_path(zh_doc, namelist)
            if zpath:
                sents = read_xml_from_zip(zf, zpath, join_char='')  # no spaces for Chinese
                if sents:
                    zh_sentences[zh_doc] = sents
    print(f"  Got {len(zh_sentences)}/{len(zh_docs_needed)} ZH documents")
    
    # ── EN: extract from remote zip via HTTP range requests ──
    print(f"\n  Extracting EN sentences from remote en.zip...")
    print(f"  (Only fetching {len(en_docs_needed)} files, not the full 24 GB)")
    
    en_sentences = {}  # en_doc_path → {sentence_id: text}
    try:
        remote = HTTPRangeFile(en_zip_url)
        with zipfile.ZipFile(remote, 'r') as zf:
            namelist = set(zf.namelist())
            found = 0
            for en_doc in en_docs_needed:
                zpath = resolve_zip_path(en_doc, namelist)
                if zpath:
                    try:
                        sents = read_xml_from_zip(zf, zpath, join_char=' ')
                        if sents:
                            en_sentences[en_doc] = sents
                            found += 1
                            if found % 20 == 0:
                                print(f"\r  Fetched {found}/{len(en_docs_needed)} EN docs...",
                                      end="", flush=True)
                    except Exception:
                        pass
            print(f"\r  Got {found}/{len(en_docs_needed)} EN documents              ")
    except Exception as e:
        print(f"  ERROR accessing remote en.zip: {e}")
        print(f"  Check your internet connection and try again.")
        return {}
    
    # ── Join using xtargets ──
    print(f"\n  Building aligned pairs from xtargets...")
    result = {}
    
    for imdb_id, doc_info in best_docs.items():
        en_sents = en_sentences.get(doc_info['en_doc'], {})
        zh_sents = zh_sentences.get(doc_info['zh_doc'], {})
        if not en_sents or not zh_sents:
            continue
        
        pairs = []
        for en_sids, zh_sids in doc_info['alignments']:
            # Concatenate multi-sentence alignments (e.g., 2 EN sents → 1 ZH sent)
            en_text = ' '.join(en_sents.get(s, '') for s in en_sids).strip()
            zh_text = ''.join(zh_sents.get(s, '') for s in zh_sids).strip()
            if en_text and zh_text:
                pairs.append({'en': en_text, 'zh': zh_text})
        
        if pairs:
            result[imdb_id] = pairs
    
    total = sum(len(v) for v in result.values())
    print(f"  {len(result)} movies, {total} aligned pairs")
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OPUS_DIR, exist_ok=True)
    
    # Check cache
    if os.path.exists(PARALLEL_CACHE) and os.path.getsize(PARALLEL_CACHE) > 10:
        with open(PARALLEL_CACHE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data:
            total = sum(len(v) for v in data.values())
            print(f"Already cached: {len(data)} movies, {total} pairs")
            print(f"Delete {PARALLEL_CACHE} to re-run extraction.")
            return
    
    # Load IMDB mapping from Step 1
    mapping_path = os.path.join(CORNELL_DIR, "imdb_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: {mapping_path} not found.")
        print("Run step1_cornell_and_imdb.py first!")
        return
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    target_imdb_ids = set(v['imdb_id'] for v in mapping.values())
    print(f"Looking for {len(target_imdb_ids)} movies from Cornell\n")
    
    # 2a: Download and parse alignment XML
    print("=" * 60)
    print("STEP 2a: Download alignment XML (119 MB)")
    print("=" * 60)
    alignment_path = os.path.join(OPUS_DIR, "en-zh_cn.xml.gz")
    download_with_progress(OPUS_ALIGNMENT_URL, alignment_path)
    
    print("\n" + "=" * 60)
    print("STEP 2b: Parse alignment XML for target movies")
    print("=" * 60)
    matched = parse_alignment_xml(alignment_path, target_imdb_ids)
    if not matched:
        print("No matched movies found!")
        return
    
    # Show coverage before big downloads
    total_docs = sum(len(v) for v in matched.values())
    print(f"\n  {len(matched)} Cornell movies found in OPUS ({total_docs} subtitle pairs)")
    
    # Reverse lookup for display
    imdb_to_title = {}
    for cid, info in mapping.items():
        imdb_to_title[info['imdb_id']] = f"{info['title']} ({info['year']})"
    
    print(f"\n  Top 15 by alignment count:")
    for iid in sorted(matched.keys(),
                      key=lambda x: -max(len(d['alignments']) for d in matched[x]))[:15]:
        best = max(len(d['alignments']) for d in matched[iid])
        title = imdb_to_title.get(iid, f"IMDB {iid}")
        print(f"    {title}: ~{best} aligned pairs")
    
    # 2c: Download zh_cn.zip
    print("\n" + "=" * 60)
    print("STEP 2c: Download ZH subtitles zip (1.6 GB)")
    print("=" * 60)
    zh_zip_path = os.path.join(OPUS_DIR, "zh_cn.zip")
    if not download_with_progress(OPUS_ZH_ZIP_URL, zh_zip_path):
        return
    
    # 2d: Extract parallel pairs
    print("\n" + "=" * 60)
    print("STEP 2d: Extract aligned EN↔ZH pairs")
    print("=" * 60)
    result = extract_parallel_pairs(matched, zh_zip_path, OPUS_EN_ZIP_URL)
    
    if not result:
        print("No parallel pairs extracted!")
        return
    
    # Cache
    with open(PARALLEL_CACHE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    
    total = sum(len(v) for v in result.values())
    print(f"\n{'=' * 60}")
    print("STEP 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Movies with parallel data: {len(result)}")
    print(f"  Total aligned EN↔ZH pairs: {total}")
    print(f"  Saved to: {PARALLEL_CACHE}")
    
    print(f"\n  Top 20 movies:")
    for iid in sorted(result.keys(), key=lambda x: -len(result[x]))[:20]:
        title = imdb_to_title.get(iid, f"IMDB {iid}")
        print(f"    {title}: {len(result[iid])} pairs")


if __name__ == "__main__":
    main()