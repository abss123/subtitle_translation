# Subtitle Translation Data Pipeline
## For: Social Power-Graph for Pragmatic Subtitle Translation (EN→ZH)

## What You're Building

You need three things for each movie:
1. **English screenplay dialogue** with speaker labels and character metadata (from Cornell)
2. **Aligned EN↔ZH subtitle pairs** where line N in English = line N in Chinese (from OPUS)
3. **IMDB IDs** to link the two sources together

## Data Sources

| Source | What it provides | Size |
|--------|-----------------|------|
| Cornell Movie-Dialogs | EN dialogue + speaker labels + character metadata | 10 MB |
| IMDb title.basics.tsv | Movie title → IMDB ID mapping | 150 MB |
| OPUS alignment XML | Which EN/ZH subtitle docs exist per movie + sentence ID mappings | 119 MB |
| OPUS zh_cn.zip | Chinese subtitle XML files | 1.6 GB |
| OPUS en.zip | English subtitle XML files | 24 GB (we don't download this fully) |

## Scripts — Run in This Order

### Script 1: `step1_cornell_and_imdb.py`
**Downloads**: Cornell corpus (10 MB) + IMDb basics (150 MB)
**Produces**: `data/cornell/` with parsed corpus + `data/cornell/imdb_mapping.json`
**Purpose**: Gets you the screenplay data with speaker labels and resolves IMDB IDs

```bash
python step1_cornell_and_imdb.py
```

### Script 2: `step2_parallel_subtitles.py`
**Downloads**: OPUS alignment XML (119 MB) + zh_cn.zip (1.6 GB) + selective EN files via HTTP range requests
**Produces**: `data/opus/parallel_en_zh.json` — aligned EN↔ZH pairs grouped by IMDB ID
**Purpose**: Gets you the translation training data

```bash
python step2_parallel_subtitles.py
```

### Script 3: `step3_build_dataset.py`
**Downloads**: Nothing (uses cached data from steps 1-2)
**Produces**: `data/combined/` — one JSON per movie with everything merged
**Purpose**: Merges Cornell screenplay data + OPUS parallel pairs by IMDB ID

```bash
python step3_build_dataset.py
```

### Utility: `lookup_movie.py`
**Purpose**: Importable module to query the dataset as DataFrames

```python
from lookup_movie import get_conversations, get_parallel_df, get_characters

df = get_parallel_df("pulp fiction")       # aligned EN↔ZH pairs
convos = get_conversations("pulp fiction") # speaker-labeled EN dialogue
chars = get_characters("pulp fiction")     # character metadata
```

## What the Final Data Looks Like

For each movie in `data/combined/{cornell_id}_{imdb_id}.json`:

```json
{
  "title": "Pulp Fiction",
  "year": "1994",
  "imdb_id": "0110912",
  "characters": {
    "u0": {"name": "JULES", "gender": "m", "credit_pos": "2"}
  },
  "conversations": [
    {
      "participants": ["JULES", "VINCENT"],
      "turns": [
        {"speaker": "JULES", "text": "You know what they call a Quarter Pounder..."}
      ]
    }
  ],
  "parallel_subtitles": [
    {"en": "You know what they call a Quarter Pounder with Cheese in Paris?",
     "zh": "你知道在巴黎他们怎么称呼四分之一磅芝士汉堡吗?"}
  ]
}
```

## How the Pieces Fit Your Project

- **`conversations`** → Build the social power graph (who speaks to whom, hierarchy, formality)
- **`parallel_subtitles`** → Train/evaluate the translation model (EN→ZH with social context)
- **`characters`** → Node attributes for the power graph (gender, credit position = importance)