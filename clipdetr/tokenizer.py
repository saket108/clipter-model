from transformers import AutoTokenizer
import torch


class SimpleTokenizer:
    """Wraps Hugging Face `AutoTokenizer` for easy use in this project.

    - model_name: pretrained tokenizer name (default: 'bert-base-uncased')
    - max_length: padding/truncation length (default: 32)
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, texts):
        """texts: list[str] -> tensor shape (B, max_length)
        Returns torch.LongTensor of input ids (no attention mask returned here).
        """
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding["input_ids"]
