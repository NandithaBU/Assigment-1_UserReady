from __future__ import annotations

import json
import logging

import click
import pandas as pd

from . import __version__
from .config import PROVIDERS, Settings, build_extractor
from .evaluation import evaluate, format_report
from .pipeline import predict_directory


def _extractor_options(f):
    opts = [
        click.option("--provider", envvar="DOCMETA_PROVIDER", default="openai",
                     type=click.Choice(PROVIDERS), show_default=True,
                     help="openai = OpenAI-compatible server (Ollama, vLLM, ...); "
                          "transformers = run a Hugging Face model locally."),
        click.option("--base-url", envvar="DOCMETA_BASE_URL",
                     help="Server URL (default http://localhost:11434/v1, i.e. Ollama)."),
        click.option("--model", envvar="DOCMETA_MODEL", help="Model name: Ollama tag, or Hugging Face id for --provider transformers."),
        click.option("--vision-model", envvar="DOCMETA_VISION_MODEL",
                     help="Model for scanned images (defaults to --model if that is set)."),
        click.option("--device-map", envvar="DOCMETA_DEVICE_MAP", default="auto",
                     show_default=True,
                     help="transformers provider only: 'auto', 'cpu' or 'cuda'."),
        click.option("--ocr", is_flag=True, envvar="DOCMETA_OCR",
                     help="OCR scans with Tesseract so a text-only model can be used."),
        click.option("--train-csv", envvar="DOCMETA_TRAIN_CSV", type=click.Path(exists=True),
                     help="train.csv with labels (enables few-shot)."),
        click.option("--train-dir", envvar="DOCMETA_TRAIN_DIR", type=click.Path(exists=True),
                     help="Folder with training documents (enables few-shot)."),
        click.option("--shots", envvar="DOCMETA_SHOTS", default=3, show_default=True,
                     help="Few-shot examples from the training set (0 = zero-shot)."),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


def _build(opts: dict):
    return build_extractor(Settings(**opts))


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True)
def main(verbose):
    """Extract metadata from agreements with an LLM."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(message)s")


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@_extractor_options
def extract(file, **opts):
    """Extract fields from one .docx / image file and print JSON."""
    result = _build(opts).extract_file(file)
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", default="outputs/predictions.csv", show_default=True)
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False),
              help="CSV with a 'File Name' column (e.g. test.csv) to restrict/order files.")
@click.option("--workers", default=2, show_default=True,
              help="Parallel requests. Keep low for a single local GPU.")
@_extractor_options
def predict(input_dir, output, manifest, workers, **opts):
    """Run extraction over a folder and write a predictions CSV."""
    extractor = _build(opts)
    df = predict_directory(extractor, input_dir, output, manifest, workers)
    click.echo(f"Wrote {len(df)} predictions to {output}")


@main.command(name="evaluate")
@click.argument("predictions", type=click.Path(exists=True, dir_okay=False))
@click.argument("truth", type=click.Path(exists=True, dir_okay=False))
@click.option("--errors", type=click.Path(), help="Write every miss to this CSV.")
@click.option("--strict", is_flag=True, help="Compare raw strings (no whitespace cleanup).")
@click.option("--ignore-case", is_flag=True)
def evaluate_cmd(predictions, truth, errors, strict, ignore_case):
    """Print per-field recall of PREDICTIONS against labelled TRUTH."""
    pred = pd.read_csv(predictions, dtype=str, keep_default_na=False)
    gold = pd.read_csv(truth, dtype=str)
    scores, misses = evaluate(pred, gold, strict=strict, ignore_case=ignore_case)
    click.echo(format_report(scores))
    if errors:
        misses.to_csv(errors, index=False)
        click.echo(f"\n{len(misses)} misses written to {errors}")


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(host, port):
    """Start the REST API (needs the 'api' extra)."""
    import uvicorn
    uvicorn.run("docmeta.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
