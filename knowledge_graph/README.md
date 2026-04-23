# Movie Scripts Knowledge Graph

This project extracts character relationships from movie dialogue with Gemini,
builds a character knowledge graph, and translates dialogue line by line into
Traditional Chinese with relationship context added to the translation prompt.

## Main Features

1. Load the Cornell Movie Dialogs Corpus.
2. Select one movie by title.
3. Reconstruct ordered dialogue turns with speaker names.
4. Chunk dialogue for LLM extraction.
5. Use Gemini to extract characters and relationships as structured JSON.
6. Merge chunk-level relations into graph-ready node and edge tables.
7. Detect the likely addressee for each translated turn using local dialogue
   context.
8. Translate each dialogue line into Traditional Chinese using the specific
   speaker-addressee relationship when available.

## Model Choice

The default model is `gemini-2.5-flash-lite`, because Google's Gemini API rate
limit table lists it as the Gemini text model with the highest free-tier daily
request allowance among the 2.5 models. You can override it with `--model`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-gemini-api-key"
```

Keep API keys in environment variables or a local `.env` file. `.env` is already
ignored by Git.

To run commands inside the project environment without changing PowerShell's
execution policy, call the interpreter directly:

```powershell
.venv\Scripts\python movie_kg_pipeline.py --movie-title "Titanic" --max-chunks 2
```

## Quick Local Validation

Run without LLM calls to validate parsing, chunking, graph export, and mock
translation structure:

```powershell
.venv\Scripts\python movie_kg_pipeline.py --movie-title "Titanic" --max-chunks 2 --max-translation-turns 20 --translate
```

## Run Gemini Extraction

```powershell
.venv\Scripts\python movie_kg_pipeline.py --movie-title "Titanic" --run-llm --max-chunks 5
```

Remove `--max-chunks` when you are ready to process the full movie.

## Run Extraction and Translation

```powershell
.venv\Scripts\python movie_kg_pipeline.py --movie-title "Titanic" --run-llm --translate --translation-batch-size 20
```

Useful throttling options:

```powershell
.venv\Scripts\python movie_kg_pipeline.py --movie-title "Titanic" --run-llm --translate --sleep-between-calls-sec 4.2
```

Focused addressee-aware translation test:

```powershell
.venv\Scripts\python movie_kg_pipeline.py --movie-title "Titanic" --run-llm --translate --max-chunks 2 --translation-start-turn 35 --max-translation-turns 12 --translation-batch-size 12 --output-dir outputs_titanic_addressee_test
```

## Outputs

By default, files are written under `outputs_cornell_gemini/`:

- `{movie_id}_turns.csv`
- `{movie_id}_nodes.csv`
- `{movie_id}_edges.csv`
- `{movie_id}_addressees.csv`
- `{movie_id}_nodes.json`
- `{movie_id}_edges.json`
- `{movie_id}_addressees.json`
- `{movie_id}_translations_zh_tw.csv`
- `{movie_id}_translations_zh_tw.json`
- `chunk_results/*.json`

## Notebook

`cornell_movie_relationship_extraction.ipynb` contains an English notebook
workflow that imports and runs the same pipeline functions.
