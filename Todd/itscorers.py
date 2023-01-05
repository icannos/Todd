from typing import Dict, List, Union

import torch
from transformers.generation_utils import ModelOutput

from .basescorers import SequenceSoftMaxScorerBase, mask_pad_tokens


class SequenceRenyiNegScorer(SequenceSoftMaxScorerBase):
    """
    Filters a batch of outputs based on the Renyi entropy of the first sequence returned for each input.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        temperature: float = 2.0,
        pad_token_id: int = 0,
        mode="input",
        num_return_sequences: int = 1,
        num_beam: int = 1,
        batch_size: int = 1,
    ):
        super().__init__(temperature, pad_token_id, mode=mode)
        self.num_beam = num_beam
        self.num_return_sequences = num_return_sequences
        self.alpha = alpha

        self.score_names = ["score"]

    def per_token_scores(
        self,
        output: ModelOutput,
    ):
        """
        :param output: ModelOutput object from huggingface generator. We need the scores and the generated sequences
        :return: a mask of size (batch_size, 1) where 0 means that the sequence is anomalous
        """
        batch_size = output.sequences.shape[0] // self.num_return_sequences

        # (num_gen_tokens, batch_size*numbeam*numreturn, vocab_size)

        # Retieve probability distribution over the vocabulary for all sequences
        # We don't keep the first score since it gives information ont the SOS token
        probabilities = self.mk_probability(torch.stack(output.scores))

        # (num_gen_tokens, batch_size*numbeam*numreturn, 1)
        # Renyi divergence against the uniform distribution
        per_step_scores = torch.log(torch.pow(probabilities, self.alpha).sum(-1))

        per_step_scores -= (self.alpha - 1) * torch.log(
            torch.ones_like(per_step_scores) * probabilities.shape[-1]
        )
        probabilities *= 1 / (self.alpha - 1)
        per_step_scores = per_step_scores.transpose(0, 1)
        per_step_scores = per_step_scores.view(
            batch_size, self.num_return_sequences, -1
        )

        return per_step_scores

    def per_output_scores(
        self,
        output: ModelOutput,
    ) -> torch.Tensor:
        # (batch_size, 1)
        # aggregate the scores over the generated tokens

        per_step_scores = self.per_token_scores(output)
        anomaly_scores = self.aggregate_step_by_step_scores(
            output.sequences,
            per_step_scores,
            self.num_return_sequences,
        )

        return anomaly_scores

    def per_input_scores(
        self,
        output: ModelOutput,
    ) -> torch.Tensor:
        # (batch_size, num_return)
        per_output_scores = self.per_output_scores(output)

        # (batch_size, 1)
        anomaly_scores = per_output_scores.mean(-1)
        return anomaly_scores

    def compute_scores_benchmark(
        self, output: ModelOutput
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :return: (*,) tensor of Mahalanobis distances
        """

        self.batch_size = output.sequences.shape[0] // self.num_return_sequences

        if self.mode == "input":
            scores = self.per_input_scores(output)
        elif self.mode == "output":
            scores = self.per_output_scores(output)
        elif self.mode == "token":
            scores = self.per_token_scores(output)
            # build mask
            mask = mask_pad_tokens(output.sequences, scores, self.pad_token_id)
            # Transform scores into list of variable length
            seq_lengths = mask.sum(-1)

            scores = scores.view(self.batch_size * self.num_return_sequences, -1)
            _scores = []
            for i in range(self.batch_size * self.num_return_sequences):
                _scores.append(scores[i, : seq_lengths[i]].tolist())
            scores = _scores

        else:
            raise ValueError(f"Unknown mode {self.mode} for {self}")

        scores = {"score": scores}

        return scores

    def fit(self, *args, **kwargs):
        pass

    def __format__(self, format_spec):
        return f"RenyiNegFilter(alpha={self.alpha}, temperature={self.temperature}, mode={self.mode})"


class BeamRenyiInformationProjection(SequenceSoftMaxScorerBase):
    def __init__(
        self,
        alpha: float = 1.5,
        temperature: float = 2.0,
        pad_token_id: int = 0,
        mode="input",
        use_soft_projection=False,
        n_neighbors=-1,
        num_return_sequences: int = 1,
        num_beams: int = 1,
    ):
        super().__init__(temperature, pad_token_id, mode=mode)
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.n_neighbors = n_neighbors
        self.use_soft_projection = use_soft_projection

        self.alpha = alpha
        self.score_names = ["score"]

    def per_output_scores(
        self,
        output: ModelOutput,
    ) -> torch.Tensor:
        # Retieve probability distribution over the vocabulary for all sequences

        self.batch_size = output.sequences.shape[0] // self.num_return_sequences
        # [len_gen, batch_size*numreturn, vocab_size]
        probabilities = self.mk_probability(torch.stack(output.scores))

        # [batch_size*numreturn, len_gen, vocab_size]
        probabilities = probabilities.transpose(0, 1)

        mask = mask_pad_tokens(output.sequences, probabilities, self.pad_token_id)
        prob_types = (probabilities * mask[:, :, None]).sum(1) / mask.sum(-1)[:, None]

        # [batch_size, numreturn, vocab_size]
        prob_types = prob_types.view(self.batch_size, self.num_return_sequences, -1)

        # [batch_size, numreturn]
        scores = self.projection_function(prob_types)

        return scores

    def projection_function(self, prob_types):
        # Returns the pair a pair divergence between types of the sequence in the beam
        # Broadcasting is used to do it in one line
        numerator = torch.pow(prob_types[:, :, None, :], self.alpha)
        denominator = torch.pow(prob_types[:, None, :, :] + 1e-20, self.alpha - 1)
        summation = (numerator / denominator).sum(-1)
        dd = (1 / (self.alpha - 1)) * torch.log(summation)

        # We are interested in the projection of each elemen onto the set of other elements
        # Obviously the min would be 0 on the diagonal since the projection of an element onto itself is 0
        # So we set it to inf to avoid it

        dd += torch.diag(torch.inf * torch.ones(dd.shape[1]))[None, :, :]

        if self.use_soft_projection:
            # We use the soft projection
            # We compute the mean divergence of the n_neighbors nearest neighbors of each element
            if self.n_neighbors == -1:
                n_neighbors = dd.shape[1] - 1
            else:
                n_neighbors = self.n_neighbors

            scores, _ = torch.topk(dd, k=n_neighbors, dim=-1, largest=False)
            scores = scores.mean(-1)
        else:
            # Otherwise we use the hard projection
            # We just take the divergence with the closest neighbor
            scores, _ = dd.min(dim=2)

        return scores

    def accumulate(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def per_input_scores(
        self,
        output: ModelOutput,
    ) -> torch.Tensor:

        per_output_scores = self.per_output_scores(output)

        return per_output_scores.mean(-1)

    def compute_scores_benchmark(
        self, output: ModelOutput
    ) -> Dict[str, Union[torch.Tensor, List]]:

        if self.mode == "input":
            scores = self.per_input_scores(output)
        elif self.mode == "output":
            scores = self.per_output_scores(output)
        elif self.mode == "token":
            raise NotImplementedError("Token mode not implemented for this filter")
        else:
            raise ValueError(f"Unknown mode {self.mode} for {self}")

        scores = {"score": scores}

        return scores

    def per_token_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("This method makes no sense for this filter")

    def __format__(self, format_spec):
        return (
            f"BeamRenyiInformationProjection(alpha={self.alpha}, "
            f"use_soft_projection={self.use_soft_projection},"
            f"n_neighbors={self.n_neighbors},"
            f"temperature={self.temperature}, mode={self.mode})"
        )
