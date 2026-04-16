"""
Translate English Movie Subtitles to Mandarin Chinese using Meta's NLLB
========================================================================
Uses: facebook/nllb-200-distilled-600M (or 1.3B for higher quality)

Input:  .srt subtitle file in English
Output: .srt subtitle file with Mandarin Chinese translations

Usage:
    python translate_subtitles.py input.srt
    python translate_subtitles.py input.srt --output output.srt
    python translate_subtitles.py input.srt --model facebook/nllb-200-1.3B
    python translate_subtitles.py input.srt --batch-size 16 --device cuda
"""

import re
import argparse
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from transformers import pipeline


# ---------------------------------------------------------------------------
# SRT parsing / writing
# ---------------------------------------------------------------------------

@dataclass
class SubtitleBlock:
    index: int
    timestamp: str
    lines: list[str]


def parse_srt(path: str) -> list[SubtitleBlock]:
    """Parse an SRT file into a list of SubtitleBlock objects."""
    text = Path(path).read_text(encoding="utf-8-sig")  # handle BOM
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    blocks = []
    for raw_block in re.split(r"\n{2,}", text.strip()):
        lines = raw_block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        timestamp = lines[1].strip()
        dialogue = lines[2:]
        blocks.append(SubtitleBlock(index=index, timestamp=timestamp, lines=dialogue))
    return blocks


def write_srt(blocks: list[SubtitleBlock], path: str) -> None:
    """Write translated SubtitleBlocks back to an SRT file."""
    parts = []
    for b in blocks:
        parts.append(str(b.index))
        parts.append(b.timestamp)
        parts.extend(b.lines)
        parts.append("")  # blank line between blocks
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def strip_html(text: str) -> str:
    """Remove basic SRT HTML tags (<i>, <b>, <u>, <font ...>) from text."""
    return re.sub(r"<[^>]+>", "", text).strip()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def build_pipeline(model_name: str, device: str):
    """Load the NLLB translation pipeline."""
    print(f"Loading model: {model_name}  (first run downloads weights)")
    translator = pipeline(
        "translation",
        model=model_name,
        device=device,
        max_length=256,
    )
    return translator


def translate_blocks(
    blocks: list[SubtitleBlock],
    translator,
    batch_size: int = 8,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "zho_Hans",
) -> list[SubtitleBlock]:
    """
    Translate all dialogue lines in the subtitle blocks.

    NLLB uses BCP-47 + script codes:
        eng_Latn  → English
        zho_Hans  → Simplified Chinese (Mandarin)
        zho_Hant  → Traditional Chinese (use --tgt-lang zho_Hant for Traditional)
    """
    # Collect (block_idx, line_idx, text) triples that need translation
    items: list[tuple[int, int, str]] = []
    for bi, block in enumerate(blocks):
        for li, line in enumerate(block.lines):
            clean = strip_html(line).strip()
            if clean:
                items.append((bi, li, clean))

    if not items:
        return blocks

    texts = [t for _, _, t in items]

    # Batch translate with progress bar
    translated_texts: list[str] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Translating", unit="batch"):
        batch = texts[start : start + batch_size]
        results = translator(
            batch,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        translated_texts.extend(r["translation_text"] for r in results)

    # Write translations back into the blocks (in-place copy)
    import copy
    translated_blocks = copy.deepcopy(blocks)
    for (bi, li, _), zh_text in zip(items, translated_texts):
        translated_blocks[bi].lines[li] = zh_text

    return translated_blocks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Translate English .srt subtitles to Mandarin using Meta NLLB"
    )
    parser.add_argument("input", help="Input English .srt file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .srt path (default: <input>_zh.srt)",
    )
    parser.add_argument(
        "--model", "-m",
        default="facebook/nllb-200-distilled-600M",
        help="HuggingFace NLLB model ID  (default: distilled-600M). "
             "Use facebook/nllb-200-1.3B for higher quality.",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Number of sentences per translation batch (default: 8)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: 'cpu', 'cuda', 'cuda:0', 'mps' (default: cpu)",
    )
    parser.add_argument(
        "--tgt-lang",
        default="zho_Hans",
        help="NLLB target language code (default: zho_Hans = Simplified). "
             "Use zho_Hant for Traditional Chinese.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or str(Path(input_path).with_suffix("")) + "_zh.srt"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # 1. Parse
    blocks = parse_srt(input_path)
    print(f"Parsed {len(blocks)} subtitle blocks")

    # 2. Load model
    translator = build_pipeline(args.model, args.device)

    # 3. Translate
    translated = translate_blocks(
        blocks,
        translator,
        batch_size=args.batch_size,
        tgt_lang=args.tgt_lang,
    )

    # 4. Write
    write_srt(translated, output_path)
    print(f"\nDone. Translated subtitles saved to: {output_path}")


if __name__ == "__main__":
    main()
