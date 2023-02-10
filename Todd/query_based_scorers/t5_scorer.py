import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Todd.query_based_scorers.query_based_scorer import QueryBasedScorer


class T5QueryScorer(QueryBasedScorer):
    def __init__(self,
                 model: T5ForConditionalGeneration = None,
                 tokenizer: T5Tokenizer = None,
                 batch_size: int = 32,
                 prefix: str = None,
                 loss_on_first_word_only: bool = True):
        super().__init__(model, tokenizer, batch_size, prefix)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.score_names = ["score"]
        self.loss_on_first_word_only = loss_on_first_word_only

    def score_sentence(self, sentence: str, model=None, tokenizer=None):
        if self.prefix is not None:
            sentence = sentence.replace(self.prefix, "")

        model = model if model is not None else self.model
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        assert isinstance(model, T5ForConditionalGeneration)

        sentences, labels = self.return_masked_input(sentence)
        loss = 0

        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            batch_labels = labels[i:i + self.batch_size]

            input_ids = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).input_ids.to(
                model.device)
            labels_ = tokenizer(batch_labels, return_tensors="pt", padding=True, truncation=True).input_ids.to(
                model.device)

            with torch.no_grad():
                lm_logits = model(input_ids=input_ids, labels=labels_).logits
                # Here we only extract the loss for the first word of the masked prediction
                if self.loss_on_first_word_only:
                    loss = max(loss, self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels_.view(-1)).view(len(labels_), -1)[:, 1].max().item())
                else:
                    loss = max(loss, self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels_.view(-1)).view(len(labels_), -1).max().item())
        return loss

    @staticmethod
    def return_masked_input(sentence: str):
        """Return the T5 input_ids and attention_mask for all combinations of 1-word sample masking"""
        words = sentence.split()
        new_sentences = []
        new_labels = []
        for i in range(len(words)):
            new_words = words.copy()
            new_words[i] = "<extra_id_0>"
            new_sentence = " ".join(new_words).strip()
            new_label = "<extra_id_0>" + words[i] + "<extra_id_1>"
            new_sentences.append(new_sentence)
            new_labels.append(new_label)

        return new_sentences, new_labels

    def __format__(self, format_spec):
        return f"T5QueryScorer(loss_on_first_word_only={self.loss_on_first_word_only})"
