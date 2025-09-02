import argparse, json
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from evaluate import load as load_metric

PREFIX = "summarize: "

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--dataset', default='cnn_dailymail')
    p.add_argument('--dataset_config', default='3.0.0')
    p.add_argument('--max_source_length', type=int, default=512)
    p.add_argument('--max_target_length', type=int, default=128)
    p.add_argument('--split', default='test')
    p.add_argument('--num_samples', type=int, default=1000)
    args = p.parse_args()

    rouge = load_metric('rouge')
    tok = T5TokenizerFast.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    ds = ds.select(range(min(args.num_samples, len(ds))))

    preds, refs = [], []
    for ex in ds:
        inp = tok(PREFIX + ex['article'], return_tensors='pt', truncation=True, max_length=args.max_source_length)
        out_ids = model.generate(**inp, max_length=args.max_target_length, min_length=32, no_repeat_ngram_size=3)
        preds.append(tok.decode(out_ids[0], skip_special_tokens=True))
        refs.append(ex['highlights'])

    res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    res = {k: round(v.mid.fmeasure * 100, 2) for k, v in res.items()}
    print(res)
    with open(f"{args.model_path}/metrics.json", 'w') as f:
        json.dump(res, f, indent=2)
