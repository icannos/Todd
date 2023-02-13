from typing import List


class QueryBasedScorer:
    def __init__(self, model=None, tokenizer=None, batch_size: int = 32, prefix: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.score = None
        self.prefix = prefix

    def score_sentence(self, sentence: str, model=None, tokenizer=None):
        raise NotImplementedError

    def score_sentences(self, sentences: List[str], model=None, tokenizer=None):
        model = model if self.model is None else self.model
        tokenizer = tokenizer if self.tokenizer is None else self.tokenizer

        return [self.score_sentence(sentence, model, tokenizer) for sentence in sentences]

    def accumulate(self, output):
        pass

    def fit(self):
        pass
