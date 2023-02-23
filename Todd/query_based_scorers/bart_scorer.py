import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from Todd.query_based_scorers.query_based_scorer import QueryBasedScorer


class BartQueryScorer(QueryBasedScorer):
    def __init__(self,
                 model: BartForConditionalGeneration = None,
                 tokenizer: BartTokenizer = None,
                 batch_size: int = 32,
                 prefix: str = None,
                 loss_on_first_word_only: bool = False,
                 repetitions: int = 10):
        super().__init__(model, tokenizer, batch_size, prefix)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.score_names = ["score"]
        self.loss_on_first_word_only = loss_on_first_word_only
        self.repetitions = repetitions

    def score_sentence(self, sentence: str, model=None, tokenizer=None):
        if self.prefix is not None:
            sentence = sentence.replace(self.prefix, "")

        assert isinstance(model, BartForConditionalGeneration)

        sentences, labels = self.return_masked_input(sentence, tokenizer)
        loss = 0

        for i in range(0, len(sentences), self.batch_size):
            with torch.no_grad():
                batch_sentences = sentences[i:i + self.batch_size].to(model.device)
                batch_labels = labels[i:i + self.batch_size].to(model.device)
                lm_logits = model(input_ids=batch_sentences, labels=batch_labels).logits

                # Here we only extract the loss for the first word of the masked prediction
                if self.loss_on_first_word_only:
                    masked_index = (batch_sentences == tokenizer.mask_token_id).nonzero()
                    loss = max(loss,
                               self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), batch_labels.view(-1)).view(len(batch_labels),-1)[
                                   masked_index[:, 0], masked_index[:, 1]].max().item())
                else:
                    loss = max(loss, self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), batch_labels.view(-1)).view(len(batch_labels), -1).max().item())
        return loss

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
        return f"BartQueryScorer(loss_on_first_word_only={self.loss_on_first_word_only})"
