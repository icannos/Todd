from transformers import LogitsProcessor
import torch
import numpy as np


class GoodTuringLogitsProcessor(LogitsProcessor):
    def __init__(self, temperature=1.0, n_boxes=100):
        self.temperature = temperature
        self.n_boxes = n_boxes

        self.bounds = torch.tensor(np.linspace(0, 0.2, n_boxes + 1))

    def __call__(self, input_ids, scores):

        proba = torch.softmax(scores / self.temperature, dim=-1)
        vocab_size = proba.shape[-1]

        box_counts = torch.zeros(proba.shape[0], self.n_boxes)
        box_masks = torch.zeros(
            proba.shape[0], self.n_boxes, vocab_size, dtype=torch.bool
        )

        for i in range(self.n_boxes):
            box_masks[:, i, :] = (proba >= self.bounds[i]) & (
                proba < self.bounds[i + 1]
            )

            box_counts[:, i] = box_masks[:, i, :].sum(dim=-1)

        print(box_masks)

        box_proba = torch.zeros((proba.shape[0], self.n_boxes))

        box_proba[:, 0] = box_counts[:, 0]

        for r in range(1, self.n_boxes):
            box_proba[:, r] = (r + 1) / ((box_counts[:, r - 1] + 1e-8))

        adjusted_proba = torch.zeros_like(proba)

        for i in range(self.n_boxes):
            adjusted_proba += box_proba[:, i, None] * box_masks[:, i, :]

        return adjusted_proba


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    vocab_size = 100

    procesor = GoodTuringLogitsProcessor(temperature=1.0, n_boxes=100)

    random_scores = torch.rand(1, vocab_size).float()

    adjusted = procesor(input_ids=None, scores=random_scores)[0]

    # Make a single plot with 2 axes with bar charts of the scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.bar(range(vocab_size), random_scores[0].softmax(dim=-1))
    ax2.bar(range(vocab_size), adjusted)
    plt.show()
