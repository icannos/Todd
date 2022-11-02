from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers.generation_utils import ModelOutput

from .basefilters import EncoderBasedFilters


def extract_batch_embeddings(
    per_layer_embeddings,
    output,
    y: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
    hidden_states="encoder_hidden_states",
) -> Tuple[Dict[Tuple[int, int], List[torch.Tensor]], torch.Tensor]:
    if layers is None:
        layers = range(len(output[hidden_states]))

    if y is None:
        y = torch.zeros(output[hidden_states][0].shape[0], dtype=torch.long)

    for layer in layers:
        # Retrieve the average embedding of the input sequence
        emb = output[hidden_states][layer].mean(dim=1)

        # Append the embeddings to the list of embeddings for the layer
        for i in range(emb.shape[0]):
            per_layer_embeddings[(layer, int(y[i]))].append(emb[i].detach().cpu())

    return per_layer_embeddings, y


def extract_embeddings(
    model, tokenizer, dataloader: DataLoader, layers: Optional[List[int]] = None
) -> Tuple[Dict[Tuple[int, int], List[torch.Tensor]], torch.Tensor]:
    """
    Extract the embeddings of the input sequences. Not classified per class.
    :param layers: List of layers to return. If None, return all layers.
    :param model: huggingface model
    :param tokenizer: huggingface tokenizer
    :param dataloader: dataloader of the input sequences
    :return: a dictionary with the embeddings of the input sequences
    """
    per_layer_embeddings = defaultdict(list)

    # Todo: take classes into account
    with torch.no_grad():
        for batch in dataloader:
            # Retrieves hidden states from the model
            inputs = tokenizer(
                batch["source"], padding=True, truncation=True, return_tensors="pt"
            )
            output = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
            )

            per_layer_embeddings, y = extract_batch_embeddings(
                per_layer_embeddings,
                output,
                layers=layers,
            )

    return per_layer_embeddings, y


class MahalanobisFilter(EncoderBasedFilters):
    """
    Filters a batch of outputs based on the Mahalanobis distance of the first sequence returned for each input.
    """

    def __init__(
        self,
        threshold: float,
        layers: List[int] = (-1,),
    ):
        super().__init__(threshold)
        self.covs = None
        self.means = None

        self.layers = set(layers)

        self.accumulated_embeddings = defaultdict(list)

    def accumulate(self, output: ModelOutput) -> None:
        """
        Accumulate the embeddings of the input sequences in the filter. To be used before fitting
        the filter with self.fit.
        :param output: Model output
        """

        per_layer_embeddings = extract_batch_embeddings(
            self.accumulated_embeddings,
            output,
            self.layers,
        )

    def fit(
        self, per_layer_embeddings: Dict[Tuple[int, int], List[torch.Tensor]], **kwargs
    ):
        """
        Prepare the filter by computing necessary statistics on the reference dataset.
        :param per_layer_embeddings:
        """
        # Compute the means and covariance matrices of the embeddings
        self.means = {
            (layer, cl): torch.stack(per_layer_embeddings[(layer, cl)]).mean(dim=0)
            for layer, cl in per_layer_embeddings.keys()
            if layer in self.layers
        }
        self.covs = {
            (layer, cl): torch.stack(per_layer_embeddings[(layer, cl)], dim=1).cov()
            for layer, cl in per_layer_embeddings.keys()
            if layer in self.layers
        }

    def dump_filter(self, path: Path):
        torch.save((self.means, self.covs), path)

    def load_filter(self, path: Path):
        self.means, self.covs = torch.load(path)

    def compute_scores(self, output: ModelOutput):
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :param layer: layer to use
        :return: (*,) tensor of Mahalanobis distances
        """
        # Retrieve mean embedding of the input sequence

        scores = defaultdict(list)

        for layer, cl in self.means.keys():
            emb = output["encoder_hidden_states"][layer].mean(dim=1)
            delta = emb - self.means[(layer, cl)]
            cov = self.covs[(layer, cl)]

            prod = torch.linalg.solve(cov[None, :, :], delta[:, :, None]).squeeze(-1)
            m = torch.bmm(delta[:, None, :], prod[:, :, None]).squeeze(-1).squeeze(-1)
            scores[layer].append(m)

        for layer, score in scores.items():
            stacked = torch.stack(score, dim=-1)
            v, _ = stacked.min(dim=-1)
            scores[layer] = v

        return torch.stack([v for k, v in scores.items()], dim=-1).mean(dim=-1)
