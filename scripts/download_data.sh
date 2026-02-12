## for locomo
mkdir -p data/locomo
cd data/locomo
wget https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
cd ../..

## for hotpotqa
mkdir -p data/hotpotqa
cd data/hotpotqa
echo "HotpotQA dataset files are not hardcoded in this repo."
echo "Set HOTPOTQA_EVAL_400_URL / HOTPOTQA_EVAL_1600_URL / HOTPOTQA_EVAL_6400_URL and re-run."
[[ -n "${HOTPOTQA_EVAL_400_URL:-}" ]] && wget "$HOTPOTQA_EVAL_400_URL"
[[ -n "${HOTPOTQA_EVAL_1600_URL:-}" ]] && wget "$HOTPOTQA_EVAL_1600_URL"
[[ -n "${HOTPOTQA_EVAL_6400_URL:-}" ]] && wget "$HOTPOTQA_EVAL_6400_URL"
cd ../..

## for ruler
python download_data/download_ruler.py

## for narrativeqa
python download_data/download_narrativeqa.py
