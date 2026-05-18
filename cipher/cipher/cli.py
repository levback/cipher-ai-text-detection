"""CIPHER CLI — AI text detection tool.

Usage
-----
cipher detect "Some text here..."
cipher detect-file report.pdf
cipher batch labels.csv
cipher train meta --data train.csv --output-dir models/
cipher train contrastive --data train.csv --output-dir models/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

from cipher.training.train_meta import train_app

app = typer.Typer(
    name="cipher",
    help="CIPHER: AI-generated text detector. Cognitive Information Pattern Hierarchy for Entropy-based Recognition.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(train_app, name="train")

console = Console()
err_console = Console(stderr=True, style="bold red")

# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

ModelDirOption = Annotated[
    Path | None,
    typer.Option("--model-dir", "-m", help="Directory with trained model weights."),
]
SpacyModelOption = Annotated[
    str,
    typer.Option("--spacy-model", help="spaCy model name (e.g. en_core_web_sm)."),
]
EncoderOption = Annotated[
    bool,
    typer.Option("--encoder/--no-encoder", help="Use sentence-transformers encoder."),
]
CoverageOption = Annotated[
    float,
    typer.Option("--coverage", min=0.80, max=0.99, help="Conformal coverage level."),
]
JsonOutputOption = Annotated[
    bool,
    typer.Option("--json", help="Output raw JSON instead of formatted display."),
]
SentenceScoresOption = Annotated[
    bool,
    typer.Option("--sentences/--no-sentences", help="Show per-sentence scores."),
]


# ---------------------------------------------------------------------------
# Shared detector factory
# ---------------------------------------------------------------------------


def _make_detector(
    model_dir: Path | None,
    spacy_model: str,
    use_encoder: bool,
    coverage: float,
) -> "CIPHERDetector":  # type: ignore[name-defined]
    from cipher.detect import CIPHERDetector

    nlp = None
    if spacy_model:
        try:
            import spacy
            nlp = spacy.load(spacy_model)
        except Exception:
            pass

    encoder = None
    if use_encoder:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass

    return CIPHERDetector(
        model_dir=model_dir,
        nlp=nlp,
        encoder=encoder,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 0.75:
        return "bright_red"
    if score >= 0.50:
        return "yellow"
    return "bright_green"


def _render_result(result: "DetectionResult", show_sentences: bool) -> None:  # type: ignore[name-defined]
    score = result.document_score
    color = _score_color(score)
    label = result.label

    # Header panel
    lo, hi = result.confidence_interval
    interval_str = f"[{lo:.2f}, {hi:.2f}]  ({result.coverage * 100:.0f}% CI)"
    console.print(
        Panel(
            f"[bold {color}]{label}[/]  [{color}]{score:.0%}[/] AI probability\n"
            f"[dim]Prediction set: {result.prediction_set}   "
            f"Interval: {interval_str}[/dim]",
            title="[bold]CIPHER Detection Result[/bold]",
            border_style=color,
            expand=False,
        )
    )

    # Signal breakdown table
    bd = result.signal_breakdown
    table = Table(title="Signal Breakdown", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Layer", style="dim")
    table.add_column("Description")
    table.add_column("Score", justify="right")

    def _fmt_score(s: float | None) -> str:
        if s is None:
            return "[dim]N/A[/dim]"
        c = _score_color(s)
        return f"[{c}]{s:.0%}[/{c}]"

    table.add_row("Layer 1", "Compression entropy", _fmt_score(bd.layer1_compression))
    table.add_row("Layer 2", "Cognitive load fingerprint", _fmt_score(bd.layer2_cognitive))
    table.add_row("Layer 3", "Discourse topology", _fmt_score(bd.layer3_discourse))
    table.add_row("Layer 4", "Contrastive manifold", _fmt_score(bd.layer4_manifold))
    console.print(table)

    # Sentence scores
    if show_sentences and result.sentence_scores:
        console.print("\n[bold]Sentence Scores[/bold]")
        for ss in result.sentence_scores[:20]:  # cap at 20 to keep output readable
            marker = "⬛" if ss.is_highlighted else "⬜"
            color = _score_color(ss.score)
            excerpt = ss.text[:80] + ("…" if len(ss.text) > 80 else "")
            console.print(
                f"  {marker} [{color}]{ss.score:.0%}[/{color}]  [dim]{excerpt}[/dim]"
            )
        if len(result.sentence_scores) > 20:
            console.print(f"  [dim]… {len(result.sentence_scores) - 20} more sentences[/dim]")

    # Metadata
    console.print(
        f"\n[dim]{result.word_count} words · {result.processing_ms:.0f} ms · "
        f"model {result.model_version}[/dim]"
    )


def _result_to_dict(result: "DetectionResult") -> dict:  # type: ignore[name-defined,type-arg]
    lo, hi = result.confidence_interval
    return {
        "document_score": result.document_score,
        "prediction_set": result.prediction_set,
        "label": result.label,
        "confidence_interval": [lo, hi],
        "coverage": result.coverage,
        "signal_breakdown": result.signal_breakdown.as_dict(),
        "sentence_scores": [
            {
                "index": ss.index,
                "text": ss.text,
                "score": ss.score,
                "is_highlighted": ss.is_highlighted,
            }
            for ss in result.sentence_scores
        ],
        "highlighted_spans": [
            {"start": sp.start, "end": sp.end, "score": sp.score}
            for sp in result.highlighted_spans
        ],
        "metadata": {
            "word_count": result.word_count,
            "processing_ms": result.processing_ms,
            "model_version": result.model_version,
        },
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def detect(
    text: Annotated[str, typer.Argument(help="Text to analyse (use '-' to read stdin).")],
    model_dir: ModelDirOption = None,
    spacy_model: SpacyModelOption = "en_core_web_sm",
    use_encoder: EncoderOption = True,
    coverage: CoverageOption = 0.95,
    show_sentences: SentenceScoresOption = True,
    as_json: JsonOutputOption = False,
) -> None:
    """Detect AI-generated content in a text string."""
    if text == "-":
        text = sys.stdin.read()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Loading models …")
        detector = _make_detector(model_dir, spacy_model, use_encoder, coverage)

    try:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
            p.add_task("Analysing …")
            result = detector.detect(text, include_sentence_scores=show_sentences)
    except ValueError as exc:
        err_console.print(str(exc))
        raise typer.Exit(1)

    if as_json:
        console.print_json(json.dumps(_result_to_dict(result)))
    else:
        _render_result(result, show_sentences)


@app.command("detect-file")
def detect_file(
    path: Annotated[Path, typer.Argument(help="Path to a .txt or .md file.")],
    model_dir: ModelDirOption = None,
    spacy_model: SpacyModelOption = "en_core_web_sm",
    use_encoder: EncoderOption = True,
    coverage: CoverageOption = 0.95,
    show_sentences: SentenceScoresOption = True,
    as_json: JsonOutputOption = False,
) -> None:
    """Detect AI-generated content in a text file."""
    if not path.exists():
        err_console.print(f"File not found: {path}")
        raise typer.Exit(1)

    text = path.read_text(encoding="utf-8", errors="replace")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
        p.add_task("Loading models …")
        detector = _make_detector(model_dir, spacy_model, use_encoder, coverage)

    try:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
            p.add_task(f"Analysing {path.name} …")
            result = detector.detect(text, include_sentence_scores=show_sentences)
    except ValueError as exc:
        err_console.print(str(exc))
        raise typer.Exit(1)

    if as_json:
        console.print_json(json.dumps(_result_to_dict(result)))
    else:
        console.print(f"[dim]File: {path}[/dim]")
        _render_result(result, show_sentences)


@app.command()
def batch(
    input_path: Annotated[Path, typer.Argument(help="CSV with columns: id,text")],
    output_path: Annotated[Path, typer.Argument(help="Output JSONL file path.")],
    model_dir: ModelDirOption = None,
    spacy_model: SpacyModelOption = "en_core_web_sm",
    use_encoder: EncoderOption = False,  # disabled by default for batch speed
    coverage: CoverageOption = 0.95,
    include_sentences: Annotated[bool, typer.Option()] = False,
) -> None:
    """Batch-detect AI content from a CSV file. Writes results as JSONL."""
    import csv

    if not input_path.exists():
        err_console.print(f"File not found: {input_path}")
        raise typer.Exit(1)

    with input_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        err_console.print("Input CSV is empty.")
        raise typer.Exit(1)

    console.print(f"Loaded {len(rows)} documents.")
    detector = _make_detector(model_dir, spacy_model, use_encoder, coverage)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with output_path.open("w") as out_fh:
        with Progress(console=console) as progress:
            task = progress.add_task("Processing …", total=len(rows))
            for row in rows:
                doc_id = row.get("id", "")
                text = row.get("text", "")
                try:
                    result = detector.detect(text, include_sentence_scores=include_sentences)
                    entry = {"id": doc_id, **_result_to_dict(result)}
                except Exception as exc:
                    entry = {"id": doc_id, "error": str(exc)}
                    errors += 1
                out_fh.write(json.dumps(entry) + "\n")
                progress.advance(task)

    console.print(f"Results written to [bold]{output_path}[/bold]  ({errors} errors)")


@app.command()
def evaluate(
    data_path: Annotated[Path, typer.Argument(help="CSV with columns: text,label (1=AI, 0=Human)")],
    model_dir: ModelDirOption = None,
    spacy_model: SpacyModelOption = "en_core_web_sm",
    use_encoder: EncoderOption = False,
    coverage: CoverageOption = 0.95,
) -> None:
    """Evaluate CIPHER on a labelled dataset and report metrics."""
    import csv

    if not data_path.exists():
        err_console.print(f"File not found: {data_path}")
        raise typer.Exit(1)

    with data_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    detector = _make_detector(model_dir, spacy_model, use_encoder, coverage)

    scores: list[float] = []
    labels: list[int] = []
    errors = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Evaluating …", total=len(rows))
        for row in rows:
            try:
                result = detector.detect(row["text"], include_sentence_scores=False)
                scores.append(result.document_score)
                labels.append(int(row["label"]))
            except Exception:
                errors += 1
            progress.advance(task)

    if not scores:
        err_console.print("No valid predictions produced.")
        raise typer.Exit(1)

    import numpy as np

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        auroc = roc_auc_score(labels_arr, scores_arr)
        ap = average_precision_score(labels_arr, scores_arr)
    except ImportError:
        auroc = ap = float("nan")

    # TPR @ FPR=1%
    thresholds = np.linspace(0, 1, 1000)
    best_tpr = 0.0
    for thresh in thresholds:
        preds = (scores_arr >= thresh).astype(int)
        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())
        tn = int(((preds == 0) & (labels_arr == 0)).sum())
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        if fpr <= 0.01:
            best_tpr = max(best_tpr, tpr)

    table = Table(title="Evaluation Results", box=box.SIMPLE_HEAVY)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("AUROC", f"{auroc:.4f}" if not np.isnan(auroc) else "N/A")
    table.add_row("AP", f"{ap:.4f}" if not np.isnan(ap) else "N/A")
    table.add_row("TPR @ FPR=1%", f"{best_tpr:.4f}")
    table.add_row("Documents evaluated", str(len(scores)))
    table.add_row("Errors skipped", str(errors))
    console.print(table)


if __name__ == "__main__":
    app()
