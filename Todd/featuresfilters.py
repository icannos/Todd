from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable

import torch
from torch.utils.data import DataLoader
from transformers.generation_utils import ModelOutput

from .basefilters import EncoderBasedFilters


def extract_batch_embeddings(
    per_layer_embeddings,
    output,
    y: Optional[torch.Tensor] = None,
    layers: Optional[Iterable[int]] = None,
    hidden_states="encoder_hidden_states",
) -> Tuple[Dict[Tuple[int, int], List[torch.Tensor]], torch.Tensor]:
    if layers is None:
        layers = range(len(output[hidden_states]))

    if y is None:
        y = torch.zeros(output[hidden_states][0].shape[0], dtype=torch.long)

    for layer in layers:
        # We use the first token embedding as representation of the sequence for now
        # It avoids the problem of the different length of the sequences when trying to take the average embedding
        emb = output[hidden_states][layer][:, 0, ...]

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

    def accumulate(self, output: ModelOutput, y: Optional[List[int]] = None) -> None:
        """
        Accumulate the embeddings of the input sequences in the filter. To be used before fitting
        the filter with self.fit.
        It is an encapsulation of extract_batch_embeddings / extract embeddings directly in the detector.
        @param output: Model output
        @param y: classes of the input sequences (used to build per class references)
        """

        per_layer_embeddings, _ = extract_batch_embeddings(
            per_layer_embeddings=self.accumulated_embeddings,
            output=output,
            layers=self.layers,
            y=y,
        )

        for key, ref_list in per_layer_embeddings.items():
            self.accumulated_embeddings[key].extend(ref_list)

    def fit(
        self,
        per_layer_embeddings: Optional[
            Dict[Tuple[int, int], List[torch.Tensor]]
        ] = None,
        **kwargs,
    ):
        """
        Prepare the filter by computing necessary statistics on the reference dataset.
        :param per_layer_embeddings:
        """

        if per_layer_embeddings is None:
            per_layer_embeddings = self.accumulated_embeddings

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

    def compute_per_layer_per_class_distances(
        self, output: ModelOutput
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """

        @param output: Huggingface model outputs with 'encoder_hidden_states'.
        @return: Dictionary of distances per layer and per class. Tensor of shape (batch_size, )
        """
        scores: Dict[Tuple[int, int], torch.Tensor] = {}

        for layer, cl in self.means.keys():
            # We take only the first token embedding as representation of the sequence for now
            # It avoids the problem of the different length of the sequences when trying to take the average embedding
            # emb : (batch_size, embedding_size)
            emb = output["encoder_hidden_states"][layer][:, 0, ...]
            delta = emb - self.means[(layer, cl)]
            cov = self.covs[(layer, cl)]

            prod = torch.linalg.solve(cov[None, :, :], delta[:, :, None]).squeeze(-1)
            m = torch.bmm(delta[:, None, :], prod[:, :, None]).squeeze(-1).squeeze(-1)

            scores[(layer, cl)] = m

        return scores

    def compute_scores(self, output: ModelOutput):
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :return: (*,) tensor of Mahalanobis distances
        """

        scores = self.compute_per_layer_per_class_distances(output)

        # We take the minimum distance over all layers and classes
        scores = torch.stack([scores[(layer, cl)] for layer, cl in scores.keys()]).min(
            dim=0
        )[0]

        return scores

    def __format__(self, format_spec):
        return f"MahalanobisFilter(layers={self.layers})"
