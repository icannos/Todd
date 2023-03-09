import torch


class ScoringAggregator(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, scores, mask):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class SumAggregator(ScoringAggregator):
    def __init__(self):
        super().__init__("SumAggregator")

    def forward(self, scores, mask):
        return scores.sum()/scores.shape[0]

   
class MeanAggregator(ScoringAggregator):
    def __init__(self):
        super().__init__("MeanAggregator")

    def forward(self, scores, mask):
        return scores.mean()


class MaxAggregator(ScoringAggregator):
    def __init__(self):
        super().__init__("MaxAggregator")

    def forward(self, scores, mask):
        return scores.max()


class MaskedSumAggregator(ScoringAggregator):
    def __init__(self):
        super().__init__("MaskedSumAggregator")

    def forward(self, scores, mask):
        return scores[mask].sum()/scores.shape[0]


class MaskedMeanAggregator(ScoringAggregator):
    def __init__(self):
        super().__init__("MaskedMeanAggregator")

    def forward(self, scores, mask):
        return scores[mask].mean()


class MaskedMaxAggregator(ScoringAggregator):
    def __init__(self):
        super().__init__("MaskedMaxAggregator")

    def forward(self, scores, mask):
        return scores[mask].max()
