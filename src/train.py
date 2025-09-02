import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from transformers.trainer_utils import IntervalStrategy
from evaluate import load as load_metric
from src.utils import TrainConfig
import nltk
nltk.download('punkt', quiet=True)

PREFIX = "summarize: "

def preprocess(batch, tokenizer, max_src, max_tgt):
    inputs = [PREFIX + x for x in batch["article"]]
    model_inputs = tokenizer(inputs, max_length=max_src, truncation=True)
    labels = tokenizer(text_target=batch["highlights"], max_length=max_tgt, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model_name_or_path', default=None)
    parser.add_argument('--dataset', default='cnn_dailymail')
    parser.add_argument('--dataset_config', default='3.0.0')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--max_source_length', type=int, default=None)
    parser.add_argument('--max_target_length', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, default=None)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    # CLI overrides
    for k, v in vars(args).items():
        if v is not None and k not in ("config", "dataset", "dataset_config", "fp16"):
            setattr(cfg, k, v)
    if args.fp16:
        cfg.fp16 = True

    tokenizer = T5TokenizerFast.from_pretrained(cfg.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model_name_or_path)

    ds = load_dataset(args.dataset, args.dataset_config)
    tokenized = ds.map(
        lambda b: preprocess(b, tokenizer, cfg.max_source_length, cfg.max_target_length),
        batched=True, remove_columns=ds["train"].column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        predict_with_generate=True,
        fp16=cfg.fp16,
        seed=cfg.seed,
        save_total_limit=2
    )

    rouge = load_metric('rouge')

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
