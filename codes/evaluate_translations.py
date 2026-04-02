"""
Evaluate Subtitle Translations
================================
Aligns three SRT files side by side into a CSV for manual or automated evaluation:
  1. English SRT (from OPUS)
  2. Mandarin SRT (from OPUS — reference/ground truth)
  3. Mandarin SRT (output of translate_subtitles.py)

Usage:
    python evaluate_translations.py en.srt zh_opus.srt zh_translated.srt
    python evaluate_translations.py en.srt zh_opus.srt zh_translated.srt --output eval.csv
    python evaluate_translations.py en.srt zh_opus.srt zh_translated.srt --threshold 500
"""

import re
import csv
import argparse
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

@dataclass
class Block:
    index: int
    timestamp: str
    start_ms: int
    end_ms: int
    text: str


def timestamp_to_ms(ts: str) -> int:
    """Convert '00:01:23,456' to milliseconds."""
    ts = ts.replace(",", ".")
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def parse_srt(path: str) -> list[Block]:
    """Parse an SRT file into a list of Blocks sorted by start time."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    blocks = []
    for chunk in re.split(r"\n{2,}", raw.strip()):
        lines = chunk.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        timestamp = lines[1].strip()
        try:
            start_str, end_str = timestamp.split("-->")
            start_ms = timestamp_to_ms(start_str.strip())
            end_ms   = timestamp_to_ms(end_str.strip())
        except Exception:
            continue
        text = " ".join(strip_html(l) for l in lines[2:] if l.strip())
        blocks.append(Block(index=index, timestamp=timestamp, start_ms=start_ms, end_ms=end_ms, text=text))

    return sorted(blocks, key=lambda b: b.start_ms)


# ---------------------------------------------------------------------------
# Matching: EN block -> best ZH text
# ---------------------------------------------------------------------------

def overlap_ms(a: Block, b: Block) -> int:
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def match_zh_to_en(en_blocks: list[Block], zh_blocks: list[Block], threshold_ms: int) -> dict[int, str]:
    """
    For each EN block, build the best ZH text:

    1. Find all ZH blocks that overlap with this EN block.
    2. If the best overlap >= threshold_ms  -> use just that one block.
       If the best overlap <  threshold_ms  -> merge ALL overlapping blocks in order.
    3. If nothing overlaps at all           -> use the ZH block with the nearest start time.
    """
    result = {}

    for en in en_blocks:
        # Step 1: find all overlapping ZH blocks
        overlapping = [zh for zh in zh_blocks if overlap_ms(en, zh) > 0]

        if overlapping:
            best = max(overlapping, key=lambda zh: overlap_ms(en, zh))

            if overlap_ms(en, best) >= threshold_ms:
                # Good match — use it alone
                result[en.index] = best.text
            else:
                # Weak match — merge all overlapping blocks in chronological order
                merged = " ".join(zh.text for zh in overlapping if zh.text)
                result[en.index] = merged

        else:
            # No overlap at all — nearest start time
            nearest = min(zh_blocks, key=lambda zh: abs(zh.start_ms - en.start_ms))
            result[en.index] = nearest.text

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Align EN + OPUS ZH + translated ZH subtitles into a CSV")
    parser.add_argument("en_srt",           help="English SRT file (from OPUS)")
    parser.add_argument("zh_opus_srt",      help="Mandarin SRT file (from OPUS, reference)")
    parser.add_argument("zh_translated_srt",help="Mandarin SRT file (output of translate_subtitles.py)")
    parser.add_argument("--output", "-o",   default=None,
                        help="Output CSV path (default: results/evaluation_<en_stem>.csv)")
    parser.add_argument("--threshold", "-t", type=int, default=100,
                        help="Overlap threshold in ms (default: 500). "
                             "Below this, all overlapping ZH blocks are merged.")
    args = parser.parse_args()

    en_blocks      = parse_srt(args.en_srt)
    zh_opus_blocks = parse_srt(args.zh_opus_srt)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = args.output or str(results_dir / f"evaluation_{Path(args.en_srt).stem}.csv")

    print(f"EN SRT:         {args.en_srt}")
    print(f"OPUS ZH SRT:    {args.zh_opus_srt}")
    print(f"Translated SRT: {args.zh_translated_srt}")
    print(f"Output CSV:     {output_path}")
    print(f"Threshold:      {args.threshold}ms")
    print()

    zh_trans_blocks = parse_srt(args.zh_translated_srt)

    print(f"  EN blocks:         {len(en_blocks)}")
    print(f"  OPUS ZH blocks:    {len(zh_opus_blocks)}")
    print(f"  Translated blocks: {len(zh_trans_blocks)}")

    zh_opus_matched   = match_zh_to_en(en_blocks, zh_opus_blocks, args.threshold)
    zh_trans_by_index = {b.index: b.text for b in zh_trans_blocks}

    rows = []
    for en in en_blocks:
        rows.append({
            "index":         en.index,
            "timestamp":     en.timestamp,
            "en_text":       en.text,
            "zh_opus":       zh_opus_matched.get(en.index, ""),
            "zh_translated": zh_trans_by_index.get(en.index, ""),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "timestamp", "en_text", "zh_opus", "zh_translated"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {output_path}")


if __name__ == "__main__":
    main()
