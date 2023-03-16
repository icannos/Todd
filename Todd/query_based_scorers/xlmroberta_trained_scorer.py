from typing import Tuple, Optional, Union
import torch

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers import XLMTokenizer, XLMWithLMHeadModel

from Todd.query_based_scorers.query_based_scorer import TrainableQueryBasedScorer


class XLMRobertaTrainedQueryScorer(TrainableQueryBasedScorer):
    def __init__(self,
                 model: XLMRobertaForMaskedLM = None,
                 tokenizer: XLMRobertaTokenizer = None,
                 batch_size: int = 32,
                 prefix: str = None,
                 suffix: str = None,
                 repetitions: int = 10,
                 **kwargs):
        super().__init__(model, tokenizer, batch_size, prefix, suffix, **kwargs)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.score_names = ["score"]
        self.repetitions = repetitions

    def prepare_sentence(self, sentence: Tuple[str, str], model=None, tokenizer=None):
        assert isinstance(model, Union[XLMRobertaForMaskedLM, XLMWithLMHeadModel])

        if isinstance(sentence, tuple) and self.concat_output:
            if self.prefix is not None:
                sentence = (sentence[0].replace(self.prefix, ""), sentence[1])
            if self.suffix is not None:
                sentence = (sentence[0].replace(self.suffix, ""), sentence[1])
        else:
            raise ValueError("Sentence must be a tuple of strings")

        sentences, labels, token_type_ids = self.return_masked_input(sentence, tokenizer)
        labels = labels.to(model.device)
        logits = []
        masked_index = (sentences == tokenizer.mask_token_id)
        for i in range(0, len(sentences), self.batch_size):
            with torch.no_grad():
                batch_sentences = sentences[i:i + self.batch_size].to(model.device)
                batch_labels = labels[i:i + self.batch_size].to(model.device)
                batch_token_type_ids = token_type_ids[i:i + self.batch_size].to(model.device) if token_type_ids is not None else None
                logits.append(model(input_ids=batch_sentences, token_type_ids=batch_token_type_ids).logits)

        logits = torch.cat(logits, dim=0)
        return logits, labels, masked_index

    def return_masked_input(self, sentence: Tuple[str, str], tokenizer, mlm_probability: float = 0.15, repetitions: Optional[int] = None):
        if repetitions is None:
            repetitions = self.repetitions

        encoding = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True)
        token_type_ids = encoding.get("token_type_ids", None)
        input_ids = encoding.input_ids
        # Repeat for variability
        input_ids = input_ids[0].repeat((repetitions, 1))     # Do not repeat for now
        if token_type_ids is not None:
            token_type_ids = token_type_ids[0].repeat((repetitions, 1))
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        input_ids[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        return input_ids, labels, token_type_ids

    def __format__(self, format_spec):
        return f"XLMRobertaTrainedQueryScorer()"
