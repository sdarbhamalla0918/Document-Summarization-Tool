import argparse
from datasets import load_dataset
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.textrank import TextRankSummarizer
from evaluate import load as load_metric

LANG = 'english'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cnn_dailymail')
    p.add_argument('--dataset_config', default='3.0.0')
    p.add_argument('--split', default='test')
    p.add_argument('--num_samples', type=int, default=500)
    p.add_argument('--max_sentences', type=int, default=3)
    args = p.parse_args()

    rouge = load_metric('rouge')
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    ds = ds.select(range(min(args.num_samples, len(ds))))

    summarizer = TextRankSummarizer(Stemmer(LANG))

    preds, refs = [], []
    for ex in ds:
        parser = PlaintextParser.from_string(ex['article'], Tokenizer(LANG))
        sentences = summarizer(parser.document, args.max_sentences)
        summary = " ".join(str(s) for s in sentences)
        preds.append(summary)
        refs.append(ex['highlights'])

    res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    print({k: round(v.mid.fmeasure * 100, 2) for k, v in res.items()})
