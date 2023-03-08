import torch


class ScoringFunction(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, logits, labels):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class CrossEntropyLoss(ScoringFunction):
    def __init__(self, reduction: str = "none", ignore_index: int = -100):
        super().__init__("CrossEntropyLoss")
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits, labels):
        return self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(len(labels), -1)


class MaxSoftmaxProbability(ScoringFunction):
    def __init__(self):
        super().__init__("MaxSoftmaxProbability")

    def forward(self, logits, labels):
        return torch.nn.functional.softmax(logits, dim=-1).max(dim=-1)[0]


class SoftmaxEntropy(ScoringFunction):
    def __init__(self):
        super().__init__("SoftmaxEntropy")

    def forward(self, logits, labels):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log(probs), dim=-1)


class RenyiDivergence(ScoringFunction):
    def __init__(self, alpha: float = 0.5):
        super().__init__(f"RenyiDivergence_{alpha}")
        self.alpha = alpha

    def forward(self, logits, labels):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # TODO: check if this is correct
        return torch.sum(probs ** self.alpha, dim=-1) / (1 - self.alpha)


class RenyiDivergenceWithReference(ScoringFunction):
    def __init__(self, alpha: float = 0.5, reference: torch.Tensor = None):
        super().__init__(f"RenyiDivergenceWithReference_{alpha}")
        self.alpha = alpha
        self.reference = reference

    def forward(self, logits, labels):
        if self.reference is None:
            return torch.ones_like(labels)*0.0
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # TODO: check if this is correct
        return torch.sum(probs ** self.alpha, dim=-1) / (1 - self.alpha) - torch.sum(self.reference ** self.alpha, dim=-1) / (1 - self.alpha)
