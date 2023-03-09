import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Todd.query_based_scorers.query_based_scorer import QueryBasedScorer


class T5QueryScorer(QueryBasedScorer):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: T5Tokenizer = None,
                 batch_size: int = 32,
                 prefix: str = None,
                 suffix: str = None,
                 repetitions: int = 10
                 ):
        super().__init__(model, tokenizer, batch_size, prefix, suffix)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.score_names = ["score"]
        self.repetitions = repetitions

    def score_sentence(self, sentence: str, model=None, tokenizer=None):
        if self.prefix is not None:
            sentence = sentence.replace(self.prefix, "")
        if self.suffix is not None:
            sentence = sentence.replace(self.suffix, "")

        model = model if model is not None else self.model
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        assert isinstance(model, T5ForConditionalGeneration)

        sentences, labels = self.return_masked_input(sentence, tokenizer)
        logits = []
        masked_index = (labels < 32000)

        for i in range(0, len(sentences), self.batch_size):
            with torch.no_grad():
                batch_sentences = sentences[i:i + self.batch_size].to(model.device)
                batch_labels = labels[i:i + self.batch_size].to(model.device)
                logits.append(model(input_ids=batch_sentences, labels=batch_labels).logits)

        logits = torch.cat(logits, dim=0)
        scores = self.score_tokens(logits, labels.to(model.device), mask=masked_index)
        return scores

    def return_masked_input(self, sentence: str, tokenizer):
        """Return the T5 input_ids and attention_mask for all combinations of 1-word sample masking"""
        input_tokens = tokenizer.tokenize(sentence)
        num_sentinels = max(1, int(len(input_tokens) * 0.15))

        inputs_total = []
        labels_total = []

        for _ in range(self.repetitions):
            # Randomly choose spans of tokens to replace with sentinel tokens
            sentinel_spans = []
            starts = sorted(np.random.choice(len(input_tokens), num_sentinels, replace=False))
            for start in starts:
                end = start + 1
                sentinel_spans.append((start, end))

            # Construct the input with sentinel tokens
            input_ids = [tokenizer.convert_tokens_to_ids(t) for t in input_tokens] + [tokenizer.eos_token_id]
            labels = []
            for i, (start, end) in enumerate(sentinel_spans):
                sentinel_token = f"<extra_id_{i}>"
                sentinel_token_id = tokenizer.convert_tokens_to_ids(sentinel_token)
                input_ids[start:end] = [sentinel_token_id]
                labels.append(sentinel_token_id)
                labels.extend(tokenizer(" ".join(input_tokens[start:end])).input_ids[:-1])

            sentinel_token = f"<extra_id_{i + 1}>"
            sentinel_token_id = tokenizer.convert_tokens_to_ids(sentinel_token)
            labels.append(sentinel_token_id)
            inputs_total.append(torch.tensor(input_ids))
            labels_total.append(torch.tensor(labels))

        inputs_total = torch.nn.utils.rnn.pad_sequence(inputs_total, batch_first=True,
                                                       padding_value=tokenizer.pad_token_id)
        labels_total = torch.nn.utils.rnn.pad_sequence(labels_total, batch_first=True,
                                                       padding_value=tokenizer.pad_token_id)
        return inputs_total, labels_total

    def __format__(self, format_spec):
        return f"T5QueryScorer()"
