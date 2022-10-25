from .basefilters import EncoderBasedFilters
import torch
from torch.utils.data import DataLoader
from transformers.generation_utils import ModelOutput
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def extract_batch_embeddings(
    per_layer_embeddings,
    output,
    y: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
    hidden_states="encoder_hidden_states",
):
    if layers is None:
        layers = range(len(output[hidden_states]))

    if y is None:
        y = torch.zeros(output[hidden_states].shape[0], dtype=torch.long)

    for layer in layers:
        # Retrieve the average embedding of the input sequence
        emb = output[hidden_states][layer].mean(dim=1)

        # Append the embeddings to the list of embeddings for the layer
        for i in range(emb.shape[0]):
            per_layer_embeddings[(layer, y[i])].append(emb[i].detach().cpu())

    return per_layer_embeddings


def extract_embeddings(
    model, dataloader: DataLoader, layers: Optional[List[int]]
) -> Dict[Tuple[int, int], List[torch.Tensor]]:
    """
    Extract the embeddings of the input sequences. Not classified per class.
    :param layers: List of layers to return. If None, return all layers.
    :param model: huggingface model
    :param dataloader: dataloader of the input sequences
    :return: a dictionary with the embeddings of the input sequences
    """
    per_layer_embeddings = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            # Retrieves hidden states from the model
            output = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
                output_hidden_states=True,
            )

            per_layer_embeddings, y = extract_batch_embeddings(
                per_layer_embeddings,
                output,
                layers,
            )

    return per_layer_embeddings


class MahalanobisFilter(EncoderBasedFilters):
    """
    Filters a batch of outputs based on the Mahalanobis distance of the first sequence returned for each input.
    """

    def __init__(
        self,
        threshold: float,
        return_scores: bool = False,
        pad_token_id: int = 0,
        layers: List[int] = (-1,),
    ):
        super().__init__(threshold)
        self.covs = None
        self.means = None

        self.layers = layers
        self.pad_token_id = pad_token_id

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

    def fit(self, per_layer_embeddings: Dict[int, List[torch.Tensor]], **kwargs):
        """
        Prepare the filter by computing necessary statistics on the reference dataset.
        :param per_layer_embeddings:
        """
        # Compute the means and covariance matrices of the embeddings
        self.means = {
            layer: torch.cat(per_layer_embeddings[layer]).mean(dim=0)
            for layer in self.layers
        }
        self.covs = {
            layer: torch.cat(per_layer_embeddings[layer]).cov() for layer in self.layers
        }

    def dump_filter(self, path: Path):
        torch.save((self.means, self.covs), path)

    def load_filter(self, path: Path):
        self.means, self.covs = torch.load(path)

    def compute_scores(self, output: ModelOutput, layer):
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :param layer: layer to use
        :return: (*,) tensor of Mahalanobis distances
        """
        # Retrieve mean embedding of the input sequence
        emb = output["encoder_hidden_states"][layer].mean(dim=1)

        # Compute the Mahalanobis distance
        return torch.sum(
            (emb - self.means[layer])
            @ torch.inverse(self.covs[layer])
            * (emb - self.means[layer]),
            dim=1,
        )
