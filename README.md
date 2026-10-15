# docmeta

Extract structured metadata from rental / lease agreements, whether they arrive as a
`.docx` or as a scanned image (`.png`, `.jpg`, ...), using **open-source LLMs that run on
your own machine or server** (via Ollama/vLLM, or loaded straight from Hugging Face). No regex or hand-written rules, and it works regardless of the
document's template.

Fields extracted: **Agreement Value, Agreement Start Date, Agreement End Date,
Renewal Notice (Days), Party One, Party Two.**

## How it works

```
.docx --python-docx--> text (paragraphs, tables, headers/footers) ----------------------\
                                                                                        +--> open-source LLM
scan  --> image sent to a vision model   (default)                                      |    (JSON-schema output,
      \-> or Tesseract OCR -> text -> text model   (--ocr) ----------------------------/     optional few-shot)
                                                                                              |
                                                       format-only normalisation (whitespace, DD.MM.YYYY, ints)
                                                                                              |
                                                                                predictions.csv / JSON / REST
```

* **Any OpenAI-compatible server.** The default backend speaks the OpenAI chat API, which
  Ollama, vLLM, LM Studio and llama.cpp's server all expose. Swap models by changing a name.
* **Structured output.** The request carries a JSON schema built from `schema.py`. If the server
  doesn't support schema-constrained output, it falls back to plain JSON and the reply is parsed
  tolerantly (code fences and chatter around the JSON are ignored). `temperature=0` for repeatable runs.
* **Two ways to read scans.** A vision model reads the image directly (default), or `--ocr` runs
  Tesseract first so a text-only model can be used.
* **Few-shot (optional).** Point it at `train.csv` + `train/` and a few labelled `.docx` examples are
  added to the prompt. With small open models this usually matters a lot.
* **Normalisation is formatting only.** It trims whitespace and lays out dates/numbers; it never
  searches the document. Dates are not calendar-validated on purpose (see data notes).
* **Provider-agnostic core.** `backends.py` defines a small `Backend` protocol. A hosted Claude
  backend is included as an option (`--provider anthropic`).

## Install

### 1. Run an open-source model (Ollama shown; vLLM / LM Studio work the same way)

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:7b-instruct     # text model, used for .docx
ollama pull qwen2.5vl:7b            # vision model, used for scans
```
A 7B-class model needs roughly 5 to 8 GB of RAM/VRAM; CPU-only works but is slow. Larger models
(for example `qwen2.5:14b-instruct`) are more accurate if your hardware allows. Check the Ollama
library for current model tags.

### 2. Install the project

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[api,dev]"          # add ",ocr" if you want --ocr (also install the Tesseract binary)
```

### Alternative: load a Hugging Face model directly (no Ollama)

```bash
pip install -e ".[api,dev,hf,ocr]"      # transformers + torch + accelerate, and Tesseract OCR
docmeta predict data/train --manifest data/train.csv --shots 0 \
    --provider transformers --model Qwen/Qwen2.5-3B-Instruct --ocr \
    -o outputs/train_predictions.csv
```

* The model is downloaded from the Hugging Face Hub on first use and cached; there is no separate
  pull step. Any instruction-tuned chat model works (for example `Qwen/Qwen2.5-7B-Instruct`,
  `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`). Gated models need
  `huggingface-cli login` first.
* **Text only.** This provider cannot read images, so scans need `--ocr` (and the Tesseract
  program installed). For a vision model, serve it with vLLM and use `--provider openai`.
* A GPU is strongly recommended. On CPU a 3B model is slow (expect up to a minute or more per
  document); try `Qwen/Qwen2.5-1.5B-Instruct` for a quick smoke test. `--device-map cpu|cuda|auto`.
* Plain transformers has no schema-constrained decoding, so the JSON is parsed from the reply;
  small models occasionally produce bad JSON, in which case that file's row is left blank
  and logged.

### Other ways to serve Hugging Face models

The default `openai` provider works with any server that speaks the OpenAI chat API, so a Hub model
can also be served with **vLLM** (`vllm serve Qwen/Qwen2.5-7B-Instruct`, Linux + GPU), **llama.cpp**
(GGUF files), or LM Studio, then used with `--base-url http://localhost:8000/v1 --model <name>`.
Hugging Face's hosted inference endpoints are also OpenAI-compatible, but that runs on their
servers rather than your own (see their docs for the URL and a token).

## Run on the assignment data

Assuming `data/` contains `train.csv`, `test.csv`, `train/`, `test/`:

```bash
# Measure accuracy on the training set (zero-shot, so no example sees its own answer)
docmeta predict data/train --manifest data/train.csv --shots 0 -o outputs/train_predictions.csv
docmeta evaluate outputs/train_predictions.csv data/train.csv --errors outputs/train_errors.csv

# Predict the test set (few-shot from the training data)  -> submission file
docmeta predict data/test --manifest data/test.csv \
    --train-csv data/train.csv --train-dir data/train \
    -o outputs/test_predictions.csv
```

If you have labels for the test set: `docmeta evaluate outputs/test_predictions.csv data/test.csv`.

Scans without a vision model: `pip install -e ".[ocr]"`, then add `--ocr` and a text model
(`--model qwen2.5:14b-instruct`).

Other commands: `docmeta extract file.docx` (JSON to stdout), `docmeta serve` (API),
`docmeta --help`. Every option has an environment variable; see `.env.example`.

## REST API

```bash
docmeta serve --port 8000            # or: uvicorn docmeta.api:app
# or Docker: API + Ollama together (pull the models once, see docker-compose.yml)
docker compose up -d
```

```bash
curl -F "file=@contract.docx" http://localhost:8000/extract
```
```json
{
  "file_name": "contract",
  "fields": {
    "agreement_value": 6500,
    "agreement_start_date": "20.05.2007",
    "agreement_end_date": "20.05.2008",
    "renewal_notice_days": 15,
    "party_one": "Antonio Levy S. Ingles, Jr. and/or Mary Rose C. Ingles",
    "party_two": "GERALDINE O. GALINATO"
  }
}
```

`GET /health` for liveness; interactive docs at `/docs`. Errors: `415` unsupported/corrupt file,
`413` file over `DOCMETA_MAX_UPLOAD_MB`, `502` model returned no usable JSON. Set
`DOCMETA_TRAIN_CSV` / `DOCMETA_TRAIN_DIR` to give the API the same few-shot examples.

## Use as a library

```python
from docmeta import build_extractor
extractor = build_extractor()                  # reads env vars
print(extractor.extract_file("contract.docx"))
```

## Configuration

| Option / env var | Meaning | Default |
|---|---|---|
| `--provider` / `DOCMETA_PROVIDER` | `openai` (Ollama/vLLM/...), `transformers` (local Hugging Face model) or `anthropic` | `openai` |
| `--base-url` / `DOCMETA_BASE_URL` | Server URL | `http://localhost:11434/v1` (Ollama) |
| `--model` / `DOCMETA_MODEL` | Model for `.docx` text (Ollama tag, or Hub id for `transformers`) | `qwen2.5:7b-instruct` / `Qwen/Qwen2.5-3B-Instruct` |
| `--vision-model` / `DOCMETA_VISION_MODEL` | Model for scans | `qwen2.5vl:7b` (or `--model` if you set only that) |
| `--ocr` / `DOCMETA_OCR` | OCR scans, use text model only | off |
| `--shots`, `--train-csv`, `--train-dir` | Few-shot examples | 3 shots when train files given |
| `--device-map` / `DOCMETA_DEVICE_MAP` | `transformers` only: `auto`, `cpu`, `cuda` | `auto` |
| `DOCMETA_API_KEY` | Key for servers that need one | none |

## Evaluation

Per-field recall = `correct / (correct + incorrect-or-missing)` on exact matches
(`evaluation.py`):

* Documents whose label for a field is empty are not scored for that field.
* Whitespace is normalised on both sides (labels contain stray trailing spaces);
  `--strict` compares raw strings, `--ignore-case` ignores case.
* `--errors file.csv` lists every miss (expected vs predicted) for error analysis.


### Results

| Field | Correct Predictions | Accuracy |
|---|:---:|---:|
| Agreement Value | 4/10 | 40.00% |
| Agreement Start Date | 4/10 | 40.00% |
| Agreement End Date | 3/10 | 30.00% |
| Renewal Notice (Days) | 1/9 | 11.11% |
| Party One | 1/10 | 10.00% |
| Party Two | 2/10 | 20.00% |
| **Macro Average** | — | **25.19%** |

## Things worth knowing about the data

Observed in the 10-row `train.csv`; they shaped the design:

* Some labelled dates are not real calendar dates (`31.11.2009`, `31.02.2011`), so dates are
  copied as written and never validated or "fixed".
* Labels have stray whitespace and the header is spelled `Aggrement`; the evaluator handles both
  and also accepts `Agreement`.
* `Renewal Notice (Days)` is empty for one document; empty labels are not scored.
* Labels are not always identical to the document text (the sample contract reads `GALINATO` with
  middle initial `O.` but is labelled `Q.`), so exact-match recall has a ceiling below 100%.
* The prompt's conventions (value = monthly rent, end date derived from a duration, 1 month = 30
  days) were inferred from a few rows. After a first run, read `train_errors.csv` and adjust
  `SYSTEM_PROMPT` in `prompts.py`.

## Project layout

```
src/docmeta/
  schema.py       fields: single source of truth (add a field here)
  readers.py      .docx -> text, image -> vision input, optional Tesseract OCR
  prompts.py      system prompt and few-shot formatting
  backends.py     OpenAI-compatible (default), Hugging Face transformers, optional Anthropic
  pipeline.py     MetadataExtractor, batch prediction
  examples.py     few-shot examples from train.csv
  normalize.py    formatting-only output cleanup
  evaluation.py   per-field recall
  cli.py / api.py command line and FastAPI service
tests/            offline unit tests (no model or key needed)
```

Run the tests with `python -m unittest discover -s tests` (or `pytest`).

## Limitations

* Small open models are less accurate than frontier hosted models, and quality depends heavily on
  the model you pick; few-shot examples and a 14B+ model help. Always check `train_errors.csv`.
* Scans: image quality drives results for both the vision path and the OCR path.
* Multi-page scans work in the backend, but the readers take one image per file; PDFs are not handled.
* The code was tested offline with fake model servers and a fake transformers pipeline (and real
  Tesseract for OCR). It has not yet been run against a live Ollama/vLLM/Hugging Face model, and the FastAPI service and `docker-compose.yml` were not
  executed in the authoring environment, so do one real end-to-end run before relying on them.
