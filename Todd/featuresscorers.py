from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from transformers.modeling_outputs import ModelOutput

from .utils.output_processing import extract_hidden_state
from .basescorers import HiddenStateBasedScorers


class MahalanobisScorer(HiddenStateBasedScorers):
    """
    Filters a batch of outputs based on the Mahalanobis distance of the first sequence returned for each input.
    """

    def __init__(
        self,
        layers: List[int] = (-1,),
        **kwargs,
    ):
        super().__init__(kwargs)
        self.covs = None
        self.means = None

        self.layers = set(layers)
        self.score_names = []

    def accumulate(self, output: ModelOutput, y: Optional[List[int]] = None) -> None:
        """
        Accumulate the embeddings of the input sequences in the scorer. To be used before fitting
        the scorer with self.fit.
        It is an encapsulation of extract_batch_embeddings / extract embeddings directly in the detector.
        @param output: Model output
        @param y: classes of the input sequences (used to build per class references)
        """

        self.extract_batch_embeddings(
            output=output,
            y=y,
        )

    def fit(
        self,
        per_layer_embeddings: Optional[
            Dict[Tuple[int, int], List[torch.Tensor]]
        ] = None,
        **kwargs,
    ):
        """
        Prepare the scorer by computing necessary statistics on the reference dataset.
        :param per_layer_embeddings:
        """

        if per_layer_embeddings is None:
            per_layer_embeddings = self.accumulated_embeddings

        # Compute the means and covariance matrices of the embeddings
        self.means = {
            (layer, cl): torch.stack(per_layer_embeddings[(layer, cl)]).float().mean(dim=0).to(self.accumulation_device)
            for layer, cl in per_layer_embeddings.keys()
            if -1 in self.layers or layer in self.layers
        }
        self.covs = {
            (layer, cl): torch.stack(per_layer_embeddings[(layer, cl)], dim=1).float().cov().to(self.accumulation_device)
            for layer, cl in per_layer_embeddings.keys()
            if -1 in self.layers or layer in self.layers
        }

        del self.accumulated_embeddings

        self.score_names = [
            f"layer_{layer}_class_{cl}" for layer, cl in self.means.keys()
        ]

    def dump_scorer(self, path: Path):
        torch.save((self.means, self.covs), path)

    def load_scorer(self, path: Path):
        self.means, self.covs = torch.load(path)

    def compute_per_layer_per_class_distances(
        self, output: ModelOutput
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """

        @param output: Huggingface model outputs with 'encoder_hidden_states'.
        @return: Dictionary of distances per layer and per class. Tensor of shape (batch_size, )
        """
        scores: Dict[Tuple[int, int], torch.Tensor] = {}

        # TODO: Will not work for decoder only models
        batch_size = output.encoder_hidden_states[0].shape[0] if "encoder_hidden_states" in output.keys() else None

        for layer, cl in self.means.keys():
            hidden_state = extract_hidden_state(output, self.chosen_state, layer)

            # We take only the first token embedding as representation of the sequence for now
            # It avoids the problem of the different length of the sequences when trying to take the average embedding
            # emb : (batch_size, embedding_size)

            if not self.use_first_token_only:
                emb = hidden_state[:, 0, ...]
            else:
                emb = hidden_state.mean(dim=1)
            delta = emb - self.means[(layer, cl)]
            cov = self.covs[(layer, cl)]

            delta = delta.float().cpu()
            cov = cov.float().cpu()     # For size purposes

            prod = torch.linalg.solve(cov[None, :, :], delta[:, :, None]).squeeze(-1)
            m = torch.bmm(delta[:, None, :], prod[:, :, None]).squeeze(-1).squeeze(-1)

            # Take max per input # TODO: choice of aggregator
            if batch_size:
                scores[(layer, cl)] = m.view(batch_size, -1).max(dim=1)[0].cpu()
            else:
                scores[(layer, cl)] = m.cpu()

        return scores

    def compute_scores(self, output: ModelOutput):
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :return: (*,) tensor of Mahalanobis distances
        """

        scores = self.compute_per_layer_per_class_distances(output)

        # We take the minimum distance over all layers and classes
        # Todo: Change this behavior: choose one particular layer or better aggregation
        scores = torch.stack([scores[(layer, cl)] for layer, cl in scores.keys()]).min(
            dim=0
        )[0]

        return scores

    def compute_scores_benchmark(
        self, output: ModelOutput
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :return: (*,) tensor of Mahalanobis distances
        """

        scores = self.compute_per_layer_per_class_distances(output)
        scores = {
            f"layer_{layer}_class_{cl}": scores[(layer, cl)]
            for layer, cl in scores.keys()
        }

        return scores

    def __format__(self, format_spec):
        return f"MahalanobisScorer(layers={self.layers},use_first_token_only={self.use_first_token_only},chosen_state={self.chosen_state})"


class CosineProjectionScorer(HiddenStateBasedScorers):
    def __init__(self, layers: List[int] = (-1,), **kwargs):
        """
        Scorer based on the cosine projection of the embeddings of the input sequences.
        :param layers: Layers to use for the computation of the scores.
        :param use_first_token_only: If True, only the first token of the sequence is used for the computation of the
        scores. If false, all the tokens are used and we use the token most OOD to compute the score for each input sequence.
        Be careful, this can be very slow and memory intensive.
        """
        super().__init__(kwargs)
        self.layers = set(layers)
        self.accumulated_embeddings = defaultdict(list)

        self.reference_embeddings: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        self.score_names = []

    def accumulate(self, output: ModelOutput, y: Optional[List[int]] = None) -> None:
        self.extract_batch_embeddings(
            output=output,
            y=y,
        )

    def fit(
        self,
        per_layer_embeddings: Optional[
            Dict[Tuple[int, int], List[torch.Tensor]]
        ] = None,
    ):

        if per_layer_embeddings is None:
            per_layer_embeddings = self.accumulated_embeddings

        for key, ref_list in per_layer_embeddings.items():
            self.reference_embeddings[key] = torch.stack(ref_list)

        # free some space since we now have stored everything in the tensor
        del self.accumulated_embeddings
        self.score_names = [f"{layer}_{cl}" for layer,cl in self.reference_embeddings.keys()]

    def compute_per_layer_per_class_disimilarity(
        self, output: ModelOutput
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """

        @param output: Huggingface model outputs with 'encoder_hidden_states'.
        @return: Dictionary of distances per layer and per class. Tensor of shape (batch_size, )
        """
        scores: Dict[Tuple[int, int], torch.Tensor] = {}

        for layer, cl in self.reference_embeddings.keys():
            hidden_state = extract_hidden_state(
                output,
                self.chosen_state,
                hidden_layer_idx=layer
            )
            dim = hidden_state.shape[-1]
            input_size = hidden_state.shape
            batch_size = output.encoder_hidden_states[layer].shape[0] if "encoder_hidden_states" in output.keys() else None

            if self.use_first_token_only:
                emb = hidden_state[:, 0, ...]
            else:
                emb = hidden_state.mean(dim=1)

            emb = emb[:, None, :].view(-1, 1, dim)
            ref = self.reference_embeddings[(layer, cl)][None, :, :].to(self.accumulation_device).view(1, -1, dim)

            # Compute the cosine similarity between the embedding and the mean
            cosine_scores = torch.nn.functional.cosine_similarity(emb, ref, dim=-1)

            # We take the max so it's an OOD score: larger => more OOD
            # The max corresponds to the closest reference embedding, so the smallest distance to the distribution
            tmp_scores = -cosine_scores.max(dim=1)[0].cpu()

            # Max over beams
            if batch_size:
                scores[(layer, cl)] = tmp_scores.view(batch_size, -1).max(dim=1)[0].cpu()
            else:
                scores[(layer, cl)] = tmp_scores.cpu()

        return scores

    def compute_scores(self, output: ModelOutput):
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :return: (*,) tensor of Mahalanobis distances
        """

        scores = self.compute_per_layer_per_class_disimilarity(output)

        # We take the minimum score over the layer
        # ie the score of the layer that is the less OOD
        # And we decide that it's the OOD score of that sample
        # Todo: Change this behavior: choose one particular layer or better aggregation
        scores = torch.stack([scores[(layer, cl)] for layer, cl in scores.keys()]).min(
            dim=0
        )[0]

        return scores

    def compute_scores_benchmark(self, output: ModelOutput) -> Dict[str, torch.Tensor]:
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :return: (*,) tensor of Mahalanobis distances
        """

        scores = self.compute_per_layer_per_class_disimilarity(output)
        scores = {f"{layer}_{cl}": scores[(layer, cl)] for layer, cl in scores.keys()}
        return scores

    def __format__(self, format_spec):
        return f"CosineProjectionScorer(layers={self.layers},use_first_token_only={self.use_first_token_only},chosen_state={self.chosen_state})"
