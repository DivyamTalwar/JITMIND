# Data Download Directory

This folder contains dataset download and preprocessing scripts used for evaluation setup.

## What This Folder Does

`download_data/` is responsible for pulling benchmark datasets from Hugging Face and writing local artifacts used by downstream evaluation scripts.

Current scripts:

- `download_narrativeqa.py`
- `download_ruler.py`

## Script Details

### `download_narrativeqa.py`

Purpose:

- downloads `deepmind/narrativeqa`
- writes each split to parquet under an output directory

Default output:

- `data/narrativeqa/`

Run:

```bash
python3 download_data/download_narrativeqa.py --output_dir data/narrativeqa
```

Expected artifacts:

- `data/narrativeqa/train.parquet`
- `data/narrativeqa/validation.parquet`
- `data/narrativeqa/test.parquet`

### `download_ruler.py`

Purpose:

- downloads a RULER dataset variant (default: `lighteval/RULER-131072-Qwen2.5-Instruct`)
- transforms each split into normalized JSONL records
- extracts structured fields such as `instruction`, `context`, and `question` for each sample

Default output:

- `data/ruler/`

Run:

```bash
python3 download_data/download_ruler.py \
  --dataset_name lighteval/RULER-131072-Qwen2.5-Instruct \
  --output_dir data/ruler
```

Expected artifacts:

- `data/ruler/<split>.jsonl` for each split present in the dataset

## Dependencies

Install required packages before running:

```bash
pip install datasets numpy pyarrow
```

If running inside the project environment:

```bash
pip install -e .
pip install datasets numpy pyarrow
```

## Operational Notes

1. Downloads may be large depending on chosen dataset variant.
2. Ensure sufficient disk space under `data/`.
3. Keep dataset versions explicit (`--dataset_name`) for reproducibility.
4. If schema changes upstream, update extraction logic in `download_ruler.py`.

## Validation Checklist

After download:

1. Confirm files exist under target output dir.
2. Sample first few rows:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('data/ruler')
for f in sorted(p.glob('*.jsonl')):
    print(f'\\n{f}:')
    with f.open('r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if i == 2:
                break
            row = json.loads(line)
            print({k: row.get(k) for k in ['instruction', 'question']})
PY
```

3. Verify split sizes are non-zero and consistent with expectations.

