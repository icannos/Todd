import torch
from transformers.generation_utils import ModelOutput

from .basefilters import SequenceSoftMaxFilterBase, mask_pad_tokens


class SequenceRenyiNegFilter(SequenceSoftMaxFilterBase):
    """
    Filters a batch of outputs based on the Renyi entropy of the first sequence returned for each input.
    """

    def __init__(
        self,
        threshold: float,
        alpha: float = 1.5,
        temperature: float = 2.0,
        pad_token_id: int = 0,
        mode="input",
    ):
        super().__init__(threshold, temperature, pad_token_id, mode=mode)
        self.alpha = alpha

    def per_token_scores(
        self,
        output: ModelOutput,
        num_return_sequences: int = 1,
        num_beam: int = 1,
        batch_size: int = 1,
    ):
        """
        :param output: ModelOutput object from huggingface generator. We need the scores and the generated sequences
        :param num_return_sequences: number of sequences returned by the model
        :param num_beam: number of beams used by the model
        :param batch_size: batch size
        :return: a mask of size (batch_size, 1) where 0 means that the sequence is anomalous
        """
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

        return per_step_scores

    def per_output_scores(
        self,
        output: ModelOutput,
        num_return_sequences: int = 1,
        num_beam: int = 1,
        batch_size: int = 1,
    ) -> torch.Tensor:
        # (batch_size, 1)
        # aggregate the scores over the generated tokens

        per_step_scores = self.per_token_scores(
            output, num_return_sequences, num_beam, batch_size
        )
        per_step_scores = per_step_scores.transpose(0, 1)

        anomaly_scores = self.aggregate_step_by_step_scores(
            output.sequences,
            per_step_scores,
            num_return_sequences,
            num_beam,
            batch_size,
        )

        return anomaly_scores

    def per_input_scores(
        self,
        output: ModelOutput,
        num_return_sequences: int = 1,
        num_beam: int = 1,
        batch_size: int = 1,
    ) -> torch.Tensor:
        # (batch_size, num_return)
        per_output_scores = self.per_output_scores(
            output, num_return_sequences, num_beam, batch_size
        )

        # (batch_size, 1)
        anomaly_scores = per_output_scores.mean(-1)
        return anomaly_scores

    def fit(self, *args, **kwargs):
        pass


class BeamRenyiInformationProjection(SequenceSoftMaxFilterBase):
    def __init__(
        self,
        threshold: float,
        alpha: float = 1.5,
        temperature: float = 2.0,
        pad_token_id: int = 0,
        mode="input",
        use_soft_projection=False,
        n_neighbors=-1,
    ):
        super().__init__(threshold, temperature, pad_token_id, mode=mode)
        self.n_neighbors = n_neighbors
        self.use_soft_projection = use_soft_projection

        self.alpha = alpha

    def per_output_scores(
        self,
        output: ModelOutput,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        batch_size: int = 1,
    ) -> torch.Tensor:
        # Retieve probability distribution over the vocabulary for all sequences

        # [len_gen, batch_size*numreturn, vocab_size]
        probabilities = self.mk_probability(torch.stack(output.scores))

        # [batch_size*numreturn, len_gen, vocab_size]
        probabilities = probabilities.transpose(0, 1)

        mask = mask_pad_tokens(output.sequences, probabilities, self.pad_token_id)
        prob_types = (probabilities * mask[:, :, None]).sum(1) / mask.sum(-1)[:, None]

        # [batch_size, numreturn, vocab_size]
        prob_types = prob_types.view(batch_size, num_return_sequences, -1)

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
        num_return_sequences: int = 1,
        num_beams: int = 1,
        batch_size: int = 1,
    ) -> torch.Tensor:

        per_output_scores = self.per_output_scores(
            output, num_return_sequences, num_beams, batch_size
        )

        return per_output_scores.mean(-1)

    def per_token_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("This method makes no sense for this filter")
