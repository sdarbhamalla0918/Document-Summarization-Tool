from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch

class T5Summarizer:
    def __init__(self, model_path: str):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

    def summarize(self, text: str, max_length=128, min_length=32):
        inp = self.tokenizer("summarize: " + text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            out_ids = self.model.generate(**inp, max_length=max_length, min_length=min_length, no_repeat_ngram_size=3)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
