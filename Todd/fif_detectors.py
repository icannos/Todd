from abc import ABC
from typing import List

import numpy as np
import torch

from .basefilters import (
    DecoderBasedFilters,
)


class StepsScoresAggregator(ABC):
    def __init__(self):
        self.reference_sequences: List[np.array] = []

    def per_token_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def accumulate(self, reference_sequence: List[np.array]):
        self.reference_sequences.extend(reference_sequence)


class OutputFunctionalIsolationForestFilter(StepsScoresAggregator, DecoderBasedFilters):
    def __init__(
        self, scorer, sample_size: int = 64, alpha=0.5, dic_number=1, ntree=64, seed=123
    ):
        super().__init__()
        self.ntree = ntree
        self.seed = seed
        self.dic_number = dic_number
        self.alpha = alpha
        self.sample_size = sample_size

        self.scorer = scorer

    def per_token_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def per_output_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, x: List[torch.Tensor]):
        raise NotImplementedError
