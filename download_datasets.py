"""
Download and assemble a large AI-text-detection dataset from public HuggingFace sources.

Sources:
  1. RAID        (liamdugan/raid)                             — human + 11 AI models, multi-domain
  2. MAGE        (yaful/MAGE)                                 — multi-model multi-domain
  3. ChatGPT-Research-Abstracts (NicolaiSivesind/...)         — arXiv abstracts vs ChatGPT
  4. AI-human-text (andythetechnerd03/AI-human-text)          — mixed community dataset
  5. HC3         (Hello-SimpleAI/HC3)                         — direct JSONL download via HTTP
  6. GPT-wiki-intro (aadityaubhat/GPT-wiki-intro)             — Wikipedia intros vs GPT-3.5
  7. Existing Gutenberg/Copilot data                          — retain

Chunking strategy
-----------------
Each source document is split into paragraph-boundary-aligned chunks of
≥100 words (see ``cipher.data.chunker``).  Every chunk carries the
``doc_id`` of its source document so that downstream train/val/test splits
can be group-aware (all chunks from the same document go to the same split,
preventing cross-document leakage).

Output columns: text, label, doc_id

Splits (configurable via --train-frac / --val-frac; remainder goes to test):
  data/full/train.csv    — 70 % of balanced corpus  (default)
  data/full/val.csv      — 15 % of balanced corpus
  data/full/test.csv     — 15 % of balanced corpus

Usage:
  # Full download, no row limit:
  python download_datasets.py
  # Limited download (faster iteration):
  python download_datasets.py --max-per-source 10000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import re
import sys
from pathlib import Path

# Add cipher package to path if not installed
sys.path.insert(0, str(Path(__file__).parent / "cipher"))

from cipher.data.chunker import chunk_document  # noqa: E402

MIN_WORDS = 100   # minimum words per chunk (paragraph-boundary-aware)
MAX_WORDS = 600   # soft cap — long docs are chunked, not truncated

EXISTING_TRAIN = Path("data/train.csv")
EXISTING_EVAL  = Path("data/eval.csv")
OUTPUT_DIR     = Path("data")

# Row type: (chunk_text, label, doc_id)
Row = tuple[str, int, str]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_valid(text: str) -> bool:
    return word_count(text) >= MIN_WORDS and len(text) >= 200


def to_chunks(text: str, doc_id: str, label: int) -> list[Row]:
    """Convert a raw document text into labelled paragraph chunks.

    Uses :func:`cipher.data.chunker.chunk_document` to split at natural
    paragraph boundaries after accumulating at least *MIN_WORDS* words.
    Each resulting chunk is returned as a ``(text, label, doc_id)`` tuple.

    Args:
        text:   Raw document text.
        doc_id: Unique identifier for the source document.
        label:  Binary label (1=AI, 0=Human).

    Returns:
        list[Row]: One or more (chunk_text, label, doc_id) triples.
    """
    chunks = chunk_document(text, min_words=MIN_WORDS)
    # Also accept the full text directly if chunking yields nothing
    if not chunks and word_count(text) >= MIN_WORDS // 2:
        chunks = [text]
    return [(c, label, doc_id) for c in chunks if is_valid(c)]


# ---------------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------------

def load_existing() -> list[Row]:
    rows: list[Row] = []
    for path in [EXISTING_TRAIN, EXISTING_EVAL]:
        if path.exists():
            with open(path) as f:
                for i, row in enumerate(csv.DictReader(f)):
                    t = clean(row.get("text", ""))
                    label = int(row.get("label", 0))
                    doc_id = row.get("doc_id", f"existing_{path.stem}_{i}")
                    rows.extend(to_chunks(t, doc_id, label))
    print(f"  {len(rows)} chunks from existing train/eval CSVs")
    return rows


# ---------------------------------------------------------------------------
# Source 1: RAID
# ---------------------------------------------------------------------------

def download_raid(max_n: int) -> list[Row]:
    """RAID benchmark — human text + 11 AI generators, multi-domain."""
    from datasets import load_dataset  # type: ignore

    rows: list[Row] = []
    doc_count = 0
    print("  RAID … ", end="", flush=True)
    try:
        # Use streaming to avoid downloading the entire dataset
        ds = load_dataset("liamdugan/raid", split="train", streaming=True)
        for item in ds:
            # Skip adversarial attacks — we want clean text only
            if item.get("attack") not in (None, "", "none"):
                continue
            gen = clean(item.get("generation", ""))
            if not is_valid(gen):
                continue
            label = 0 if item.get("model", "").lower() == "human" else 1
            doc_id = f"raid_{doc_count}"
            rows.extend(to_chunks(gen, doc_id, label))
            doc_count += 1
            if len(rows) >= max_n:
                break
        ai_n = sum(1 for _, l, _ in rows if l == 1)
        print(f"{len(rows)} chunks from {doc_count} docs ({ai_n} AI / {len(rows)-ai_n} human)")
    except Exception as exc:
        print(f"SKIP ({exc})")
    return rows


# ---------------------------------------------------------------------------
# Source 2: MAGE
# ---------------------------------------------------------------------------

def download_mage(max_n: int) -> list[Row]:
    """MAGE multi-model detection benchmark."""
    from datasets import load_dataset  # type: ignore

    rows: list[Row] = []
    doc_count = 0
    print("  MAGE … ", end="", flush=True)
    try:
        ds = load_dataset("yaful/MAGE", split="train", streaming=True)
        for item in ds:
            t = clean(item.get("text", ""))
            if not is_valid(t):
                continue
            doc_id = f"mage_{doc_count}"
            rows.extend(to_chunks(t, doc_id, int(item.get("label", 0))))
            doc_count += 1
            if len(rows) >= max_n:
                break
        ai_n = sum(1 for _, l, _ in rows if l == 1)
        print(f"{len(rows)} chunks from {doc_count} docs ({ai_n} AI / {len(rows)-ai_n} human)")
    except Exception as exc:
        print(f"SKIP ({exc})")
    return rows


# ---------------------------------------------------------------------------
# Source 3: ChatGPT Research Abstracts
# ---------------------------------------------------------------------------

def download_abstracts(max_n: int) -> list[Row]:
    """ArXiv real abstracts vs ChatGPT-generated ones."""
    from datasets import load_dataset  # type: ignore

    rows: list[Row] = []
    doc_count = 0
    print("  ChatGPT-Research-Abstracts … ", end="", flush=True)
    try:
        ds = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts",
                          split="train", streaming=True)
        for item in ds:
            real = clean(item.get("real_abstract", ""))
            fake = clean(item.get("generated_abstract", ""))
            if is_valid(real):
                rows.extend(to_chunks(real, f"abstracts_real_{doc_count}", 0))
            if is_valid(fake):
                rows.extend(to_chunks(fake, f"abstracts_ai_{doc_count}", 1))
            doc_count += 1
            if len(rows) >= max_n:
                break
        ai_n = sum(1 for _, l, _ in rows if l == 1)
        print(f"{len(rows)} chunks from {doc_count} docs ({ai_n} AI / {len(rows)-ai_n} human)")
    except Exception as exc:
        print(f"SKIP ({exc})")
    return rows


# ---------------------------------------------------------------------------
# Source 4: andythetechnerd03/AI-human-text
# ---------------------------------------------------------------------------

def download_ai_human_text(max_n: int) -> list[Row]:
    """Community AI vs human text detection dataset."""
    from datasets import load_dataset  # type: ignore

    rows: list[Row] = []
    doc_count = 0
    print("  AI-human-text … ", end="", flush=True)
    try:
        ds = load_dataset("andythetechnerd03/AI-human-text",
                          split="train", streaming=True)
        for item in ds:
            t = clean(item.get("text", ""))
            label = int(item.get("generated", 0))
            if is_valid(t):
                rows.extend(to_chunks(t, f"aih_{doc_count}", label))
                doc_count += 1
            if len(rows) >= max_n:
                break
        ai_n = sum(1 for _, l, _ in rows if l == 1)
        print(f"{len(rows)} chunks from {doc_count} docs ({ai_n} AI / {len(rows)-ai_n} human)")
    except Exception as exc:
        print(f"SKIP ({exc})")
    return rows


# ---------------------------------------------------------------------------
# Source 5: HC3 via direct HTTP (JSONL files on HuggingFace)
# ---------------------------------------------------------------------------

HC3_FILES = [
    "open_qa", "wiki_csai", "finance", "medicine", "reddit_eli5",
]
HC3_BASE = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main"


def download_hc3(max_n: int) -> list[Row]:
    """HC3 Human-ChatGPT Comparison Corpus — direct JSONL download."""
    import urllib.request

    rows: list[Row] = []
    doc_count = 0
    print("  HC3 (direct JSONL) …")
    for subset in HC3_FILES:
        if len(rows) >= max_n:
            break
        url = f"{HC3_BASE}/{subset}.jsonl"
        print(f"    {subset} … ", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            # Handle gzip if served compressed
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            added = 0
            for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                for ans in item.get("human_answers", []):
                    t = clean(ans)
                    if is_valid(t):
                        rows.extend(to_chunks(t, f"hc3_{subset}_{doc_count}", 0))
                        doc_count += 1
                        added += 1
                for ans in item.get("chatgpt_answers", []):
                    t = clean(ans)
                    if is_valid(t):
                        rows.extend(to_chunks(t, f"hc3_{subset}_ai_{doc_count}", 1))
                        doc_count += 1
                        added += 1
            print(f"{added} docs")
        except Exception as exc:
            print(f"SKIP ({exc})")
    ai_n = sum(1 for _, l, _ in rows if l == 1)
    print(f"  HC3 total: {len(rows)} chunks ({ai_n} AI / {len(rows)-ai_n} human)")
    return rows


# ---------------------------------------------------------------------------
# Source 6: GPT-wiki-intro (Wikipedia intros vs GPT-3.5)
# ---------------------------------------------------------------------------

def download_gpt_wiki_intro(max_n: int) -> list[Row]:
    """Wikipedia intro paragraphs vs GPT-3.5-turbo completions."""
    from datasets import load_dataset  # type: ignore

    rows: list[Row] = []
    doc_count = 0
    print("  GPT-wiki-intro … ", end="", flush=True)
    try:
        ds = load_dataset("aadityaubhat/GPT-wiki-intro",
                          split="train", streaming=True)
        for item in ds:
            human = clean(item.get("wiki_intro", ""))
            ai    = clean(item.get("generated_intro", ""))
            if is_valid(human):
                rows.extend(to_chunks(human, f"wiki_human_{doc_count}", 0))
            if is_valid(ai):
                rows.extend(to_chunks(ai, f"wiki_ai_{doc_count}", 1))
            doc_count += 1
            if len(rows) >= max_n:
                break
        ai_n = sum(1 for _, l, _ in rows if l == 1)
        print(f"{len(rows)} chunks from {doc_count} docs ({ai_n} AI / {len(rows)-ai_n} human)")
    except Exception as exc:
        print(f"SKIP ({exc})")
    return rows


# ---------------------------------------------------------------------------
# Deduplication + balancing + save
# ---------------------------------------------------------------------------

def deduplicate(rows: list[Row]) -> list[Row]:
    """Remove duplicate chunks based on the first 120 characters of text.

    Args:
        rows: List of (text, label, doc_id) triples.

    Returns:
        list[Row]: Deduplicated list preserving first-seen order.
    """
    seen: set[str] = set()
    out: list[Row] = []
    for text, label, doc_id in rows:
        key = text[:120].lower()
        if key not in seen:
            seen.add(key)
            out.append((text, label, doc_id))
    return out


def write_csv(path: Path, rows: list[Row]) -> None:
    """Write a list of (text, label, doc_id) rows to a CSV file.

    Args:
        path: Output file path.
        rows: List of (text, label, doc_id) triples.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text", "label", "doc_id"])
        for text, label, doc_id in rows:
            writer.writerow([text, label, doc_id])


def split_and_save(
    all_rows: list[Row],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:
    """Shuffle *all_rows* and write three group-aware CSV splits.

    Groups all chunks from the same ``doc_id`` into the same partition so
    that no document straddles two splits (no cross-document leakage).
    Splits are stored under data/full/ as train.csv, val.csv, test.csv.
    The test fraction is ``1 - train_frac - val_frac``.

    Args:
        all_rows:    List of (text, label, doc_id) triples.
        seed:        Random seed for reproducible shuffling.
        train_frac:  Proportion of data for training (default 0.70).
        val_frac:    Proportion for validation (default 0.15).
    """
    from collections import defaultdict

    assert train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1"
    random.seed(seed)

    # Group rows by doc_id
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in all_rows:
        groups[row[2]].append(row)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    n_total = len(all_rows)
    n_train_target = int(n_total * train_frac)
    n_val_target   = int(n_total * val_frac)

    train_rows: list[Row] = []
    val_rows: list[Row] = []
    test_rows: list[Row] = []

    for key in group_keys:
        group = groups[key]
        if len(train_rows) < n_train_target:
            train_rows.extend(group)
        elif len(val_rows) < n_val_target:
            val_rows.extend(group)
        else:
            test_rows.extend(group)

    out = OUTPUT_DIR / "full"
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "train.csv", train_rows)
    write_csv(out / "val.csv",   val_rows)
    write_csv(out / "test.csv",  test_rows)

    # Also write a combined raw file for manual re-sampling
    write_csv(OUTPUT_DIR / "full_combined.csv", all_rows)

    def stats(rows: list[Row], name: str) -> None:
        ai  = sum(1 for _, l, _ in rows if l == 1)
        hum = len(rows) - ai
        doc_ids = {doc_id for _, _, doc_id in rows}
        print(f"  {name:6s}: {len(rows):>8,} chunks / {len(doc_ids):,} docs  ({ai:,} AI / {hum:,} human)")

    print(f"\n{'='*60}")
    stats(train_rows, "Train")
    stats(val_rows,   "Val")
    stats(test_rows,  "Test")
    n_total_actual = len(train_rows) + len(val_rows) + len(test_rows)
    n_docs_total = len({row[2] for row in all_rows})
    print(f"  {'Total':6s}: {n_total_actual:>8,} chunks / {n_docs_total:,} docs")
    print(f"  Output : data/full/{{train,val,test}}.csv + data/full_combined.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the dataset downloader.

    Parse arguments, download all sources in order, deduplicate, balance,
    and write 70/15/15 train/val/test splits to data/full/.
    """
    parser = argparse.ArgumentParser(
        description="Download AI-text-detection datasets and create train/val/test splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full download (may take 20-30 min):\n"
            "  python download_datasets.py\n"
            "  # Capped download for quick testing:\n"
            "  python download_datasets.py --max-per-source 5000\n"
        ),
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        metavar="N",
        help="Maximum rows to collect from each HuggingFace source. "
             "Defaults to None (unlimited — download full dataset).",
    )
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Omit the original Gutenberg/Copilot data.")
    parser.add_argument("--train-frac",    type=float, default=0.70,
                        help="Training split fraction (default 0.70).")
    parser.add_argument("--val-frac",      type=float, default=0.15,
                        help="Validation split fraction (default 0.15).")
    args = parser.parse_args()

    # Use a sentinel so each download function knows whether to cap
    cap = args.max_per_source if args.max_per_source is not None else 10_000_000

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_rows: list[Row] = []

    print("[1/7] Existing Gutenberg/Copilot data")
    if not args.skip_existing:
        all_rows.extend(load_existing())

    print("[2/7] RAID benchmark")
    all_rows.extend(download_raid(cap))

    print("[3/7] MAGE multi-model benchmark")
    all_rows.extend(download_mage(cap))

    print("[4/7] ChatGPT Research Abstracts")
    all_rows.extend(download_abstracts(cap))

    print("[5/7] AI-human-text (community dataset)")
    all_rows.extend(download_ai_human_text(cap))

    print("[6/7] HC3 direct JSONL download")
    all_rows.extend(download_hc3(cap))

    print("[7/7] GPT-wiki-intro")
    all_rows.extend(download_gpt_wiki_intro(cap))

    print(f"\nDeduplicating {len(all_rows):,} chunks …")
    all_rows = deduplicate(all_rows)

    ai_rows    = [r for r in all_rows if r[1] == 1]
    human_rows = [r for r in all_rows if r[1] == 0]
    print(f"  After dedup: {len(ai_rows):,} AI / {len(human_rows):,} human")

    # Keep all data without artificial cap so every downloaded sample is used
    random.shuffle(ai_rows)
    random.shuffle(human_rows)
    balanced = ai_rows + human_rows
    print(f"  Combined:    {len(ai_rows):,} AI / {len(human_rows):,} human = {len(balanced):,} total")

    split_and_save(balanced, seed=args.seed,
                   train_frac=args.train_frac, val_frac=args.val_frac)


if __name__ == "__main__":
    main()
