from abc import ABC
from typing import Callable

import torch

from Todd import ScorerType


class Filter(ABC):
    def __init__(self, scorer: ScorerType, threshold: float = 0.0):
        self.scorer = scorer
        self.threshold = threshold

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Fit the filter on the data.
        """

        self.scorer.fit(*args, **kwargs)


    def predict(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the score of the input sequences and return a mask tensor with True for the sequences to keep.
        It relies on the `compute_score` method that should be implemented by the child class.
        """
        scores = self.decision_function(*args, **kwargs)

        return scores >= 0
    def decision_function(self, *args, **kwargs) -> torch.Tensor:
        """
        Should be defined using the scoring function of the filter such that negative values are outliers/anomalies
        """
        return self.threshold - self.scorer.compute_scores(*args, **kwargs)

    def scores(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the raw scores of the input.
        """

        return self.compute_scores(*args, **kwargs)

    def compute_scores(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the raw scores of the input.
        """

        raise NotImplementedError
