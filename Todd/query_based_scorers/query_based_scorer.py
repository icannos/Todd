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
                 *args,
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefix = prefix
        self.suffix = suffix
        if scoring_functions:
            self.scoring_functions = scoring_functions
        else:
            self.scoring_functions = [CrossEntropyLoss(),
                                      SoftmaxEntropy(),
                                      MaxSoftmaxProbability(),
                                      RenyiDivergence(),
                                      RenyiDivergenceWithReference()]
        if scoring_aggregator:
            self.scoring_aggregator = scoring_aggregator
        else:
            self.scoring_aggregator = [MeanAggregator(),
                                       MaxAggregator(),
                                       MaskedMaxAggregator(),
                                       MaskedMeanAggregator(),
                                       MaskedSumAggregator(),
                                       SumAggregator()]

    def score_tokens(self, logits, labels, mask=None):
        scores = []
        for scoring_function in self.scoring_functions:
            token_scores = scoring_function(logits, labels)
            for aggregator in self.scoring_aggregator:
                agg_score = aggregator(token_scores, mask).item()
                scores.append({f"{scoring_function.name}_{aggregator.name}": agg_score})
        return scores

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
