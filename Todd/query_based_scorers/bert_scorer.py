import torch
from transformers import BertTokenizer, BertForMaskedLM
from Todd.query_based_scorers.query_based_scorer import QueryBasedScorer


class BertQueryScorer(QueryBasedScorer):
    def __init__(self,
                 model: BertForMaskedLM = None,
                 tokenizer: BertTokenizer = None,
                 batch_size: int = 32,
                 prefix: str = None,
                 suffix: str = None,
                 repetitions: int = 10):
        super().__init__(model, tokenizer, batch_size, prefix, suffix)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.score_names = ["score"]
        self.repetitions = repetitions

    def score_sentence(self, sentence: str, model=None, tokenizer=None):
        if self.prefix is not None:
            sentence = sentence.replace(self.prefix, "")
        if self.suffix is not None:
            sentence = sentence.replace(self.suffix, "")

        assert isinstance(model, BertForMaskedLM)

        sentences, labels = self.return_masked_input(sentence, tokenizer)
        logits = []
        masked_index = (sentences == tokenizer.mask_token_id)
        for i in range(0, len(sentences), self.batch_size):
            with torch.no_grad():
                batch_sentences = sentences[i:i + self.batch_size].to(model.device)
                batch_labels = labels[i:i + self.batch_size].to(model.device)
                logits.append(model(input_ids=batch_sentences, labels=batch_labels).logits)

        logits = torch.cat(logits, dim=0)
        scores = self.score_tokens(logits, labels, mask=masked_index)
        return scores

    def return_masked_input(self, sentence: str, tokenizer, mlm_probability: float = 0.15):
        input_ids = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True).input_ids
        # Repeat for variability
        input_ids = input_ids[0].repeat((self.repetitions, 1))
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        input_ids[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        return input_ids, labels

    def __format__(self, format_spec):
        return f"BertQueryScorer()"
