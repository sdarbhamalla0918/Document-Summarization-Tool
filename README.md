# Document Summarization Tool (T5)

Abstractive summarization using a fine‑tuned T5 model. Includes:
- **Training** on news datasets (CNN/DailyMail default via 🤗 Datasets)
- **Evaluation** with ROUGE-1/2/L + length & speed metrics
- **Baseline**: TextRank extractive summarizer for comparison
- **Real‑time API**: FastAPI endpoint `/summarize`
- **Docker** for reproducible serving

## Quickstart

### 1) Environment
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train (T5‑small by default)
```bash
python -m src.train   --model_name_or_path t5-small   --dataset cnn_dailymail   --dataset_config 3.0.0   --output_dir runs/t5-small-cnn   --max_source_length 512 --max_target_length 128   --per_device_train_batch_size 8 --per_device_eval_batch_size 8   --learning_rate 3e-4 --num_train_epochs 3   --gradient_accumulation_steps 2 --fp16
```

> Tip: Swap to `t5-base` or `t5-large` if you have the GPU.

### 3) Evaluate
```bash
python -m src.evaluate   --model_path runs/t5-small-cnn   --dataset cnn_dailymail --dataset_config 3.0.0
```
This prints ROUGE-1/2/L and saves a JSON report under `runs/.../metrics.json`.

### 4) Compare with Extractive Baseline (TextRank)
```bash
python -m src.baseline_extractive   --dataset cnn_dailymail --dataset_config 3.0.0   --max_sentences 3
```
Outputs ROUGE metrics for the baseline to compare.

### 5) Real‑time API
Start the API using a trained (or pre‑trained) checkpoint:
```bash
MODEL_PATH=runs/t5-small-cnn uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Then POST:
```bash
curl -X POST http://localhost:8000/summarize   -H 'Content-Type: application/json'   -d '{"text": "<your long article text>", "max_length": 128, "min_length": 32}'
```

### 6) Docker (serving)
```bash
docker build -t document-summarization-tool:latest .
docker run -p 8000:8000 -e MODEL_PATH=/model document-summarization-tool:latest
```

## Results (Template)
Replace with your actual numbers after training/eval:

| Model            | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------------------|---------|---------|---------|
| TextRank (3 sent)| 36.5    | 14.2    | 33.8    |
| T5-small (ours)  | 40.9    | 18.5    | 38.7    |
| T5-base (ours)   | 42.7    | 19.9    | 40.1    |

> **Note**: Run `src.evaluate` to generate your own metrics. Hardware, seed, and training time affect results.

## Config
See `config.yaml` for defaults. CLI args override config.

## Citation
If you use this repo, please cite the T5 paper and Hugging Face Transformers/Datasets.

## License
MIT
