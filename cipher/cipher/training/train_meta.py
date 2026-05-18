"""Training pipeline for the contrastive manifold encoder (Layer 4)
and the meta-learner (Layer 5).

Commands
--------
cipher train contrastive  --data-dir <dir>  --output-dir <dir>
cipher train meta         --data-dir <dir>  --model-dir <dir>

Chunked training with doc_id
-----------------------------
When the CSV contains a ``doc_id`` column, all chunks from the same
document are forced into the same split (group-aware stratified split).
This prevents data leakage where chunks from the same document appear
in both training and calibration sets.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

train_app = typer.Typer(help="Train CIPHER model components.", no_args_is_help=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_labeled_csv(path: Path) -> tuple[list[str], list[int], list[str]]:
    """Read a CSV with columns ``text,label[,doc_id]``.

    Args:
        path: Path to the labelled CSV file.

    Returns:
        Tuple of (texts, labels, doc_ids).  When the CSV has no ``doc_id``
        column, doc_ids are synthesised as ``"doc_{row_index}"``.
    """
    texts: list[str] = []
    labels: list[int] = []
    doc_ids: list[str] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            texts.append(row["text"])
            labels.append(int(row["label"]))
            doc_ids.append(row.get("doc_id", f"doc_{i}"))
    return texts, labels, doc_ids


def _group_aware_split(
    indices: list[int],
    doc_ids: list[str],
    val_fraction: float,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split *indices* into train/val ensuring no doc straddles the boundary.

    All indices sharing a ``doc_id`` are placed in the same split.  Groups
    are shuffled before assignment so the split is randomised.

    Args:
        indices:      Row indices to split.
        doc_ids:      doc_id for each row (parallel to the full dataset).
        val_fraction: Target fraction for the validation set.
        seed:         Random seed.

    Returns:
        (train_indices, val_indices): Two disjoint lists of row indices.
    """
    # Group indices by doc_id
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        groups[doc_ids[idx]].append(idx)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_val_target = int(len(indices) * val_fraction)
    val_indices: list[int] = []
    train_indices: list[int] = []

    for key in group_keys:
        if len(val_indices) < n_val_target:
            val_indices.extend(groups[key])
        else:
            train_indices.extend(groups[key])

    return train_indices, val_indices


def _extract_features(
    texts: list[str],
    nlp: object | None,
    encoder: object | None,
    ai_corpus: str | None,
    human_corpus: str | None,
) -> np.ndarray:
    """Run the full feature pipeline (Layers 1-3 + 6) on a list of texts.

    Args:
        texts:        List of document strings.
        nlp:          Optional spaCy model for discourse and lexical features.
        encoder:      Optional sentence-transformers encoder for cognitive features.
        ai_corpus:    Optional reference AI corpus string for NCD computation.
        human_corpus: Optional reference human corpus string for NCD computation.

    Returns:
        np.ndarray: Feature matrix of shape (len(texts), n_features).
    """
    from cipher.layers.assembler import FeatureVector
    from cipher.layers.cognitive import compute_cognitive_features
    from cipher.layers.compression import compute_compression_features
    from cipher.layers.discourse import compute_discourse_features
    from cipher.layers.lexical import compute_lexical_features

    rows: list[np.ndarray] = []
    for i, text in enumerate(texts):
        comp = compute_compression_features(text, ai_corpus, human_corpus)
        cog = compute_cognitive_features(text, encoder=encoder)
        disc = compute_discourse_features(text, nlp)
        lex = compute_lexical_features(text, nlp=nlp)
        fv = FeatureVector(compression=comp, cognitive=cog, discourse=disc, lexical=lex)
        rows.append(fv.to_array(include_manifold=False))
        if (i + 1) % 50 == 0:
            typer.echo(f"  Processed {i + 1}/{len(texts)} documents …")
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# train meta
# ---------------------------------------------------------------------------


@train_app.command("meta")
def train_meta(
    data_path: Annotated[Path, typer.Option("--data", help="CSV file (text,label[,doc_id])")],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Directory to save models")],
    spacy_model: Annotated[str, typer.Option(help="spaCy model name")] = "en_core_web_sm",
    use_encoder: Annotated[bool, typer.Option(help="Use sentence-transformers encoder")] = True,
    calibration_split: Annotated[float, typer.Option(help="Fraction held out for calibration")] = 0.2,
    coverage: Annotated[float, typer.Option(help="Conformal coverage level")] = 0.95,
) -> None:
    """Train the meta-learner (Layer 5) from a labelled CSV dataset.

    When the CSV contains a ``doc_id`` column, the calibration split is
    performed at the *document* level so that all chunks from the same
    source document appear in the same partition (no cross-doc leakage).
    """
    typer.echo(f"Loading data from {data_path} …")
    texts, labels, doc_ids = _load_labeled_csv(data_path)
    typer.echo(f"  {len(texts)} documents, {sum(labels)} AI / {len(labels) - sum(labels)} human")

    has_doc_ids = any(not d.startswith("doc_") for d in doc_ids)
    if has_doc_ids:
        n_docs = len(set(doc_ids))
        typer.echo(f"  {n_docs} unique source documents detected — using group-aware split.")

    # Load optional heavy dependencies
    nlp = None
    try:
        import spacy
        nlp = spacy.load(spacy_model)
        typer.echo(f"  spaCy model '{spacy_model}' loaded.")
    except Exception:
        typer.echo("  spaCy not available — discourse features will be partial.", err=True)

    encoder = None
    if use_encoder:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            typer.echo("  sentence-transformers encoder loaded.")
        except Exception:
            typer.echo("  sentence-transformers not available — using BoW fallback.", err=True)

    # Group-aware or simple stratified train/calibration split
    indices = list(range(len(texts)))
    if has_doc_ids:
        train_idx, cal_idx = _group_aware_split(
            indices, doc_ids, val_fraction=calibration_split
        )
    else:
        random.seed(42)
        random.shuffle(indices)
        split = int(len(indices) * (1 - calibration_split))
        train_idx, cal_idx = indices[:split], indices[split:]

    typer.echo(f"Extracting features for {len(train_idx)} training documents …")
    X_train = _extract_features(
        [texts[i] for i in train_idx], nlp, encoder, None, None
    )
    y_train = np.array([labels[i] for i in train_idx])

    typer.echo(f"Extracting features for {len(cal_idx)} calibration documents …")
    X_cal = _extract_features(
        [texts[i] for i in cal_idx], nlp, encoder, None, None
    )
    y_cal = np.array([labels[i] for i in cal_idx])

    # Train meta-learner
    from cipher.models.meta_learner import MetaLearner

    typer.echo("Training meta-learner (GBM ensemble) …")
    meta = MetaLearner()
    meta.fit(X_train, y_train)

    # Calibrate
    from cipher.calibration.isotonic import IsotonicCalibrator

    raw_cal_scores = np.array([meta.predict_proba(x) for x in X_cal])
    iso = IsotonicCalibrator()
    iso.fit(raw_cal_scores, y_cal)

    cal_scores = iso.transform_array(raw_cal_scores)

    from cipher.calibration.conformal import ConformalPredictor

    cp = ConformalPredictor(alpha=1.0 - coverage)
    cp.calibrate(cal_scores, y_cal)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    meta.save(output_dir / "meta_learner.pkl")
    iso.save(output_dir / "isotonic.pkl")
    cp.save(output_dir / "conformal.npy")

    # Report feature importances
    importances = meta.feature_importances()
    if importances:
        sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
        typer.echo("\nTop 10 feature importances:")
        for name, coef in sorted_feats[:10]:
            typer.echo(f"  {name:25s}  {coef:+.4f}")

    typer.echo(f"\nModels saved to {output_dir}")


# ---------------------------------------------------------------------------
# train contrastive  (requires torch + transformers)
# ---------------------------------------------------------------------------


@train_app.command("contrastive")
def train_contrastive(
    data_path: Annotated[Path, typer.Option("--data", help="CSV file (text,label)")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option()] = 10,
    batch_size: Annotated[int, typer.Option()] = 32,
    lr: Annotated[float, typer.Option()] = 2e-5,
    margin: Annotated[float, typer.Option()] = 0.3,
    max_seq_len: Annotated[int, typer.Option()] = 512,
) -> None:
    """Train the DeBERTa contrastive encoder (Layer 4). Requires torch."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer
    except ImportError:
        typer.echo("torch and transformers are required: pip install cipher-detect[neural]")
        raise typer.Exit(1)

    from cipher.models.contrastive_encoder import ContrastiveEncoder, TripletLoss  # pragma: no cover

    typer.echo(f"Loading data from {data_path} …")  # pragma: no cover
    texts, labels = _load_labeled_csv(data_path)  # pragma: no cover

    ai_texts = [t for t, l in zip(texts, labels) if l == 1]  # pragma: no cover
    human_texts = [t for t, l in zip(texts, labels) if l == 0]  # pragma: no cover
    typer.echo(f"  {len(ai_texts)} AI / {len(human_texts)} human texts")  # pragma: no cover

    device = "cuda" if torch.cuda.is_available() else "cpu"  # pragma: no cover
    typer.echo(f"  Training on {device}")  # pragma: no cover

    tokenizer = AutoTokenizer.from_pretrained(ContrastiveEncoder.MODEL_NAME)  # pragma: no cover
    model = ContrastiveEncoder(pretrained=True).to(device)  # pragma: no cover
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # pragma: no cover
    criterion = TripletLoss(margin=margin)  # pragma: no cover

    class TripletDataset(Dataset):  # type: ignore[type-arg]  # pragma: no cover
        def __init__(self) -> None:
            self.n = min(len(human_texts), len(ai_texts)) * 2

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict[str, str]:
            import random
            anchor = random.choice(human_texts)
            positive = random.choice(human_texts)
            negative = random.choice(ai_texts)
            return {"anchor": anchor, "positive": positive, "negative": negative}

    def collate(batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:  # pragma: no cover
        for key in ["anchor", "positive", "negative"]:
            enc = tokenizer(
                [b[key] for b in batch],
                max_length=max_seq_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            if key == "anchor":
                a_ids, a_mask = enc["input_ids"], enc["attention_mask"]
            elif key == "positive":
                p_ids, p_mask = enc["input_ids"], enc["attention_mask"]
            else:
                n_ids, n_mask = enc["input_ids"], enc["attention_mask"]
        return {
            "a_ids": a_ids, "a_mask": a_mask,
            "p_ids": p_ids, "p_mask": p_mask,
            "n_ids": n_ids, "n_mask": n_mask,
        }

    loader = DataLoader(TripletDataset(), batch_size=batch_size, collate_fn=collate)  # pragma: no cover

    for epoch in range(1, epochs + 1):  # pragma: no cover
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            za = model.encode(batch["a_ids"].to(device), batch["a_mask"].to(device))
            zp = model.encode(batch["p_ids"].to(device), batch["p_mask"].to(device))
            zn = model.encode(batch["n_ids"].to(device), batch["n_mask"].to(device))
            loss = criterion(za, zp, zn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        typer.echo(f"  Epoch {epoch}/{epochs}  loss={total_loss / len(loader):.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)  # pragma: no cover
    model.save(output_dir / "contrastive_encoder.pt")  # pragma: no cover

    # Fit GMM manifold
    typer.echo("Fitting GMM manifold …")  # pragma: no cover
    model.eval()  # pragma: no cover

    def embed_texts(tx: list[str]) -> np.ndarray:  # pragma: no cover
        embs = []
        for text in tx:
            enc = tokenizer(text, max_length=max_seq_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            with torch.no_grad():
                z = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            embs.append(z.cpu().numpy()[0])
        return np.vstack(embs)

    ai_embs = embed_texts(ai_texts[:2000])  # pragma: no cover
    human_embs = embed_texts(human_texts[:2000])  # pragma: no cover

    from cipher.models.contrastive_encoder import ManifoldScorer  # pragma: no cover

    scorer = ManifoldScorer()  # pragma: no cover
    scorer.fit(ai_embs, human_embs)  # pragma: no cover
    scorer.save(output_dir / "manifold_scorer.pkl")  # pragma: no cover

    typer.echo(f"Contrastive encoder saved to {output_dir}")  # pragma: no cover
