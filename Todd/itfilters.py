import torch
from transformers.generation_utils import ModelOutput

from .basefilters import SequenceSoftMaxFilterBase


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
    ):
        super().__init__(threshold, temperature, pad_token_id)
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
        probabilities = self.mk_probability(torch.stack(output.scores))
        # Get uniform distribution over the vocabulary
        U = torch.ones_like(probabilities) / probabilities.shape[-1]

        # (num_gen_tokens, batch_size*numbeam*numreturn, 1)
        # Renyi divergence against the uniform distribution
        per_step_scores = torch.log(
            torch.sum(U**self.alpha * probabilities ** (1 - self.alpha), dim=-1)
        ) / (self.alpha - 1)

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

        anomaly_scores = self.aggregate_step_by_step_scores(
            output.sequences,
            per_step_scores,
            num_return_sequences,
            num_beam,
            batch_size,
        )

        return anomaly_scores

    def fit(self, *args, **kwargs):
        pass


class BeamRenyiInformationProjection(SequenceSoftMaxFilterBase):
    # todo: implement this
    def per_output_scores(
        self,
        output: ModelOutput,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        batch_size: int = 1,
    ) -> torch.Tensor:

        # Retieve probability distribution over the vocabulary for all sequences
        probabilities = self.mk_probability(torch.stack(output.scores))
        # Get uniform distribution over the vocabulary
        U = torch.ones_like(probabilities) / probabilities.shape[-1]

        # (num_gen_tokens, batch_size*numbeam*numreturn, 1)
        # Renyi divergence against the uniform distribution
        per_step_scores = torch.log(
            torch.sum(U**self.alpha * probabilities ** (1 - self.alpha), dim=-1)
        ) / (self.alpha - 1)

        # batch_size*num_beams*num_return_sequences, num_gen_tokens
        per_step_scores = (
            per_step_scores.squeeze(-1).transpose(0, 1).view(batch_size, num_beams, -1)
        )
