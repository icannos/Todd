import torch
from typing import List
from Todd.query_based_scorers.scoring_functions import ScoringFunction, MaxSoftmaxProbability, CrossEntropyLoss, SoftmaxEntropy, RenyiDivergence, RenyiDivergenceWithReference
from Todd.query_based_scorers.scoring_aggregators import ScoringAggregator, MeanAggregator, MaxAggregator, MaskedMaxAggregator, MaskedMeanAggregator, MaskedSumAggregator, SumAggregator


class QueryBasedScorer:
    def __init__(self,
                 model=None,
                 tokenizer=None,
                 batch_size: int = 32,
                 prefix: str = None,
                 suffix: str = None,
                 scoring_functions: List[ScoringFunction] = None,
                 scoring_aggregator: List[ScoringAggregator] = None,
                 device: str = None,
                 *args,
                 **kwargs):

        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device) if model is not None else None
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefix = prefix
        self.suffix = suffix
        self.reference_probs = []
        self.concat_output = False

        if scoring_functions:
            self.scoring_functions = scoring_functions
        else:
            self.scoring_functions = [CrossEntropyLoss(),
                                      SoftmaxEntropy(),
                                      MaxSoftmaxProbability(),
                                      RenyiDivergence(),
                                      RenyiDivergenceWithReference(alpha=0.5),
                                      RenyiDivergenceWithReference(alpha=0.9),
                                      RenyiDivergenceWithReference(alpha=2.0),
                                      RenyiDivergenceWithReference(alpha=3.0),
                                      RenyiDivergenceWithReference(alpha=5.0),
                                      RenyiDivergenceWithReference(alpha=10.0),
                                      RenyiDivergenceWithReference(alpha=20.0)]
        if scoring_aggregator:
            self.scoring_aggregator = scoring_aggregator
        else:
            self.scoring_aggregator = [MeanAggregator(),
                                       MaxAggregator(),
                                       MaskedMaxAggregator(),
                                       MaskedMeanAggregator(),
                                       # MaskedSumAggregator(),
                                       # SumAggregator()
                                       ]

    def score_tokens(self, logits, labels, mask=None):
        scores = []
        for scoring_function in self.scoring_functions:
            token_scores = scoring_function(logits, labels)
            for aggregator in self.scoring_aggregator:
                if mask.sum() == 0:
                    agg_score = 0
                else:
                    agg_score = aggregator(token_scores, mask).fillna(0).item()
                scores.append({f"{scoring_function.name}_{aggregator.name}": agg_score})
        return scores

    def score_sentences(self, sentences: List[str], model=None, tokenizer=None):
        model = model if self.model is None else self.model
        tokenizer = tokenizer if self.tokenizer is None else self.tokenizer
        return [self.score_sentence(sentence, model, tokenizer) for sentence in sentences]

    def score_sentence(self, sentence: str, model=None, tokenizer=None):
        logits, labels, masked_index = self.prepare_sentence(sentence, model, tokenizer)
        scores = self.score_tokens(logits, labels.to(model.device), mask=masked_index)
        return scores

    def prepare_sentence(self, sentence: str, model=None, tokenizer=None):
        raise NotImplementedError

    def accumulate(self, sentences: List[str], model=None, tokenizer=None):
        model = model if self.model is None else self.model
        tokenizer = tokenizer if self.tokenizer is None else self.tokenizer

        for sentence in sentences:
            logits, labels, masked_index = self.prepare_sentence(sentence, model, tokenizer)
            self.reference_probs.append(logits[masked_index].softmax(dim=1).mean(dim=0).cpu())

    def fit(self):
        self.reference_probs = torch.stack(self.reference_probs).mean(dim=0)
        for scoring_func in self.scoring_functions:
            if isinstance(scoring_func, RenyiDivergenceWithReference):
                scoring_func.reference = self.reference_probs

    def return_masked_input(self, sentence: str, tokenizer):
        raise NotImplementedError
