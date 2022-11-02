from abc import ABC
from typing import TypeVar

import torch
from transformers.generation_utils import ModelOutput


def mean_score_remove_padding(
    sequences: torch.Tensor, scores: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    """
    Computes the mean score of a sequence of tokens, removing the padding tokens.
    :param sequences: (*, seq_len) tensor of token ids
    :param scores: (*, seq_len) tensor of scores
    :param pad_token_id: id of the padding token
    :return: (*,) tensor of mean scores
    """

    # Todo: weird check
    # Sometime scores and sequences gen size are different and i have no idea why
    if sequences.shape[1] != scores.shape[1]:
        mask = sequences[:, :-1] != pad_token_id
    else:
        mask = sequences != pad_token_id

    return ((scores * mask.float()).sum(dim=-1) / mask.sum(dim=-1).float()).squeeze()


class Filter(ABC):
    def __init__(self, threshold):
        """
        :param threshold: threshold to use for the filter
        """
        self.threshold = threshold

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the score of the input sequences and return a mask tensor with True for the sequences to keep.
        It relies on the `compute_score` method that should be implemented by the child class.
        """
        scores = self.decision_function(*args, **kwargs)

        return scores >= 0

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def accumulate(self, *args, **kwargs):
        raise NotImplementedError

    def decision_function(self, *args, **kwargs) -> torch.Tensor:
        """
        Should be defined using the scoring function of the filter such that negative values are outliers/anomalies
        """
        return self.threshold - self.compute_scores(*args, **kwargs)

    def scores(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the raw scores of the input.
        """

        return self.compute_scores(*args, **kwargs)

    def compute_scores(self, *args, **kwargs) -> torch.Tensor:
        """
        Should return an anomaly score: a higher score means more likely to be an anomaly.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


FilterType = TypeVar("FilterType", bound=Filter)


class EncoderBasedFilters(Filter):
    def __init__(self, threshold):
        super().__init__(threshold)


class DecoderBasedFilters(Filter):
    def __init__(self, threshold, mode: str = "input"):
        super().__init__(threshold)
        self.mode = mode

    def compute_scores(self, *args, **kwargs) -> torch.Tensor:
        try:
            if self.mode == "input":
                return self.per_input_scores(*args, **kwargs)
            elif self.mode == "output":
                return self.per_output_scores(*args, **kwargs)
            elif self.mode == "token":
                return self.per_token_scores(*args, **kwargs)
            else:
                raise ValueError(
                    f"Invalid mode {self.mode}. Should be one of ['input', 'output', 'token']"
                )
        except NotImplementedError as e:
            raise NotImplementedError(
                f"per_{self.mode}_scores is not implemented in {self.__class__.__name__}. "
                f"Maybe it is a bug or maybe you are using a filter that does not support this mode."
            ) from e

    def per_token_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def per_output_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def per_input_scores(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LikelyhoodFilter(DecoderBasedFilters):
    """
    Filters a batch of output based on the likelyhood of the first sequence returned for each input.
    """

    def __init__(self, threshold: float = 0.9, mode="input"):
        super().__init__(threshold, mode=mode)

    def per_output_scores(
        self, output: ModelOutput, num_return_sequences: int = 1
    ) -> torch.Tensor:
        sequences_scores = output.sequences_scores
        sequences_scores = sequences_scores.view(-1, num_return_sequences)

        return sequences_scores

    def per_input_scores(
        self, output: ModelOutput, num_return_sequences: int = 1
    ) -> torch.Tensor:
        # bs, num_return_sequences
        per_output_scores = self.per_output_scores(output.num_return_sequences)

        # todo: add option to change aggregation function
        # bs
        return per_output_scores[:, 0]


class SequenceSoftMaxFilterBase(DecoderBasedFilters):
    def __init__(
        self,
        threshold: float,
        temperature: float = 2.0,
        pad_token_id: int = 0,
        mode="input",
    ):
        super().__init__(threshold, mode=mode)
        self.pad_token_id = pad_token_id
        self.temperature = temperature
        self.threshold = threshold

    def mk_probability(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores / self.temperature, dim=-1)

    def aggregate_step_by_step_scores(
        self,
        sequences: torch.Tensor,
        per_step_scores: torch.Tensor,
        num_return_sequences: int,
        num_beam: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        :param sequences: (batch_size*numreturn, seq_len) tensor of token ids
        :param per_step_scores: (batch_size*numreturn, seq_len) tensor of scores
        :param num_return_sequences: number of sequences returned by the model
        :param num_beam: number of beams used by the model
        :param batch_size: batch size
        :return: (batch_size, 1) tensor of aggregated scores
        """
        # (batch_size*numbeam*numreturn, nun_gen_steps)
        per_step_scores = per_step_scores.squeeze(-1)

        # (batch_size*numbeam*numreturn, 1)
        anomaly_scores = mean_score_remove_padding(
            sequences, per_step_scores, self.pad_token_id
        )

        # (batch_size, numreturn)
        anomaly_scores = anomaly_scores.view(batch_size, num_return_sequences)

        return anomaly_scores


class SequenceMSPFilter(SequenceSoftMaxFilterBase):
    """
    Compute the Maximum Softmax Probability score of the input
    sequences and return a mask tensor with True for the sequences to keep.
    """

    def __init__(
        self,
        threshold: float,
        temperature: float = 2.0,
        pad_token_id: int = 0,
    ):
        super().__init__(threshold, temperature, pad_token_id)

    def per_token_scores(
        self,
        output: ModelOutput,
        num_return_sequences: int = 1,
        num_beam: int = 1,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Returns OOD scores per generated token based on the probability distribution they have been generated from.
        @param output: ModelOutput object.
        @param num_return_sequences: number of sequences returned by the model.
        @param num_beam: number of beams used by the model
        @param batch_size: batch size used during generation.
        @return: (batch_size, num_return_sequences, seq_len) tensor of scores.
        """

        sequences = output.sequences
        probabilities = self.mk_probability(output.scores)
        per_step_scores = torch.max(probabilities, dim=-1)

        return per_step_scores.view(batch_size, num_return_sequences, -1)

    def per_output_scores(
        self,
        output: ModelOutput,
        num_return_sequences: int = 1,
        num_beam: int = 1,
        batch_size: int = 1,
    ) -> torch.Tensor:
        sequences = output.sequences
        probabilities = self.mk_probability(output.scores)
        per_step_scores = torch.max(probabilities, dim=-1)

        return self.aggregate_step_by_step_scores(
            sequences, per_step_scores, num_return_sequences, num_beam, batch_size
        )
