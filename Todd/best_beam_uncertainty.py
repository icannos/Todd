from typing import Dict, List, Union

import numpy as np
import torch

from .basescorers import (
    SequenceSoftMaxScorerBase,
    mask_pad_tokens,
)

from .utils.output_processing import (
    GenerateOutputType,
)


class BestBeamSeqRenyi(SequenceSoftMaxScorerBase):
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
        output: GenerateOutputType,
    ):
        """
        :param output: ModelOutput object from huggingface generator. We need the scores and the generated sequences
        :return: a mask of size (batch_size, 1) where 0 means that the sequence is anomalous
        """
        batch_size = output.sequences.shape[0] // self.num_return_sequences

        # (num_gen_tokens, batch_size*numbeam*numreturn, vocab_size)

        # Retrieve probability distribution over the vocabulary for all sequences
        # We don't keep the first score since it gives information ont the SOS token

        # (num_return_sequences*batch_size, num_gen_tokens, vocab_size)
        scores = torch.stack(output.scores).transpose(0, 1)
        probabilities = self.mk_probability(scores)

        # If alpha is 1, we compute the KL divergence
        if self.alpha == 1:
            per_step_scores = torch.log(probabilities) * probabilities
            per_step_scores = per_step_scores.sum(-1)
            per_step_scores += torch.log(
                torch.ones_like(per_step_scores).to(probabilities.device)
                * probabilities.shape[-1]
            )
        else:
            # (num_gen_tokens, batch_size*numbeam*numreturn, 1)
            # Renyi divergence against the uniform distribution
            per_step_scores = torch.log(torch.pow(probabilities, self.alpha).sum(-1))

            per_step_scores -= (self.alpha - 1) * torch.log(
                torch.ones_like(per_step_scores) * probabilities.shape[-1]
            )
            probabilities *= 1 / (self.alpha - 1)

        per_step_scores = per_step_scores.transpose(0, 1)
        per_step_scores = per_step_scores.reshape(
            batch_size, self.num_return_sequences, -1
        )

        per_step_scores = per_step_scores.reshape(
            batch_size, self.num_return_sequences, -1
        )

        return per_step_scores

    def per_output_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        # (batch_size, 1)
        # aggregate the scores over the generated tokens

        per_step_scores = self.per_token_scores(output)
        anomaly_scores = self.aggregate_step_by_step_scores(
            output.sequences.cpu(),
            per_step_scores.cpu(),
            self.num_return_sequences,
        )

        return anomaly_scores

    def per_input_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        # (batch_size, num_return)
        per_output_scores = self.per_output_scores(output)

        # (batch_size, 1)
        anomaly_scores = per_output_scores.mean(-1)
        return anomaly_scores

    def compute_scores_benchmark(
        self, output: GenerateOutputType
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
            scores = scores.reshape(self.batch_size * self.num_return_sequences, -1)
            # build mask
            mask = mask_pad_tokens(output.sequences, scores, self.pad_token_id)
            # Transform scores into list of variable length

            seq_lengths = mask.sum(-1)

            scores = scores.view(self.batch_size, self.num_return_sequences, -1)
            _scores = []

            for i in range(self.batch_size):
                _scores.append(
                    [
                        scores[
                            i, j, : seq_lengths[i * self.num_return_sequences + j]
                        ].tolist()
                        for j in range(self.num_return_sequences)
                    ]
                )
            scores = _scores

        else:
            raise ValueError(f"Unknown mode {self.mode} for {self}")

        scores = {"score": scores}

        return scores

    def __format__(self, format_spec):
        return f"{self.__class__.__name__}(alpha={self.alpha}, temperature={self.temperature}, mode={self.mode})"


class BestBeamSeqFisherRao(SequenceSoftMaxScorerBase):
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
        output: GenerateOutputType,
    ):
        """
        :param output: ModelOutput object from huggingface generator. We need the scores and the generated sequences
        :return: a mask of size (batch_size, 1) where 0 means that the sequence is anomalous
        """
        batch_size = output.sequences.shape[0] // self.num_return_sequences

        # (num_gen_tokens, batch_size*numbeam*numreturn, vocab_size)

        # Retrieve probability distribution over the vocabulary for all sequences
        # We don't keep the first score since it gives information ont the SOS token

        # (num_return_sequences*batch_size, num_gen_tokens, vocab_size)
        scores = torch.stack(output.scores).transpose(0, 1)
        probabilities = self.mk_probability(scores)

        # Compute the Fisher-Rao divergence

        per_step_scores = torch.arccos(
            probabilities.sum(-1) * np.sqrt(probabilities.shape[-1])
        )
        per_step_scores *= 2 / torch.pi

        per_step_scores = per_step_scores.transpose(0, 1)
        per_step_scores = per_step_scores.reshape(
            batch_size, self.num_return_sequences, -1
        )

        per_step_scores = per_step_scores.reshape(
            batch_size, self.num_return_sequences, -1
        )

        return per_step_scores

    def per_output_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        # (batch_size, 1)
        # aggregate the scores over the generated tokens

        per_step_scores = self.per_token_scores(output)
        anomaly_scores = self.aggregate_step_by_step_scores(
            output.sequences.cpu(),
            per_step_scores.cpu(),
            self.num_return_sequences,
        )

        return anomaly_scores

    def per_input_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        # (batch_size, num_return)
        per_output_scores = self.per_output_scores(output)

        # (batch_size, 1)
        anomaly_scores = per_output_scores.mean(-1)
        return anomaly_scores

    def compute_scores_benchmark(
        self, output: GenerateOutputType
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
            scores = scores.reshape(self.batch_size * self.num_return_sequences, -1)
            # build mask
            mask = mask_pad_tokens(output.sequences, scores, self.pad_token_id)
            # Transform scores into list of variable length

            seq_lengths = mask.sum(-1)

            scores = scores.view(self.batch_size, self.num_return_sequences, -1)
            _scores = []

            for i in range(self.batch_size):
                _scores.append(
                    [
                        scores[
                            i, j, : seq_lengths[i * self.num_return_sequences + j]
                        ].tolist()
                        for j in range(self.num_return_sequences)
                    ]
                )
            scores = _scores

        else:
            raise ValueError(f"Unknown mode {self.mode} for {self}")

        scores = {"score": scores}

        return scores

    def __format__(self, format_spec):
        return f"{self.__class__.__name__}(alpha={self.alpha}, temperature={self.temperature}, mode={self.mode})"


class BestBeamSoftMaxEnergyScorer(SequenceSoftMaxScorerBase):
    """
    Compute the Maximum Softmax Probability score of the input
    sequences and return a mask tensor with True for the sequences to keep.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        pad_token_id: int = 0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        mode="input",
    ):
        super().__init__(temperature, pad_token_id, mode=mode)

        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.score_names = ["score"]

    def per_token_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        """
        Returns OOD scores per generated token based on the probability distribution they have been generated from.
        @param output: ModelOutput object.
        @return: (batch_size, num_return_sequences, seq_len) tensor of scores.
        """

        batch_size = output.sequences.shape[0] // self.num_return_sequences

        scores = torch.stack(output.scores).transpose(0, 1)
        scores = -self.temperature * torch.exp(scores / self.temperature).sum(-1)

        return scores.view(batch_size, self.num_return_sequences, -1)

    def per_output_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        sequences = output.sequences

        per_step_scores = self.per_token_scores(output)

        return self.aggregate_step_by_step_scores(
            sequences, per_step_scores, self.num_return_sequences
        )

    def per_input_scores(
        self,
        output: GenerateOutputType,
        num_return_sequences: int = 1,
        num_beam: int = 1,
    ) -> torch.Tensor:
        return self.per_output_scores(output)[:, 0]

    def __format__(self, format_spec):
        return f"{self.__class__.__name__}(mode={self.mode}, temperature={self.temperature}, mode={self.mode})"


class BestBeamMSPScorer(SequenceSoftMaxScorerBase):
    """
    Compute the Maximum Softmax Probability score of the input
    sequences and return a mask tensor with True for the sequences to keep.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        pad_token_id: int = 0,
        mode="input",
    ):
        super().__init__(temperature, pad_token_id, mode=mode)
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.score_names = ["score"]

    def per_token_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        """
        Returns OOD scores per generated token based on the probability distribution they have been generated from.
        @param output: ModelOutput object.
        @return: (batch_size, num_return_sequences, seq_len) tensor of scores.
        """

        batch_size = output.sequences.shape[0] // self.num_return_sequences

        scores = torch.stack(output.scores).transpose(0, 1)

        probabilities = self.mk_probability(scores)

        per_step_scores, _ = torch.max(probabilities, dim=-1)

        return per_step_scores.view(batch_size, self.num_return_sequences, -1)

    def per_output_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        sequences = output.sequences

        per_step_scores = self.per_token_scores(
            output,
        )

        return self.aggregate_step_by_step_scores(
            sequences, per_step_scores, self.num_return_sequences
        )

    def per_input_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        return self.per_output_scores(
            output,
        )[:, 0]

    def __format__(self, format_spec):
        return f"{self.__class__.__name__}(mode={self.mode}, temperature={self.temperature}, mode={self.mode})"


class BestBeamInformationProjection(SequenceSoftMaxScorerBase):
    def __init__(
        self,
        alpha: float = 1.5,
        temperature: float = 2.0,
        pad_token_id: int = 0,
        mode="output",
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

        # batch_size, voc_size
        self.stored_types = []

        self.alpha = alpha
        self.score_names = ["score"]
        self.stored_types_tensor = None

    def accumulate(self, output: GenerateOutputType):
        # (batch_size, num_return)

        self.batch_size = output.sequences.shape[0] // self.num_return_sequences
        # [len_gen, batch_size*numreturn, vocab_size]

        scores = torch.stack(output.scores).transpose(0, 1)

        probabilities = self.mk_probability(scores)
        vocab_size = probabilities.shape[-1]

        mask = mask_pad_tokens(output.sequences, probabilities, self.pad_token_id)

        prob_types = (probabilities * mask[:, :, None]).sum(1) / mask.sum(-1)[:, None]

        # [batch_size, numreturn, vocab_size]
        prob_types = prob_types.view(
            self.batch_size, self.num_return_sequences, vocab_size
        )[:, 0, :]

        self.stored_types.append(prob_types.detach().cpu())

    def fit(self, *args, **kwargs):
        self.stored_types_tensor = torch.cat(self.stored_types, 0)

        for tensor in self.stored_types:
            del tensor
        self.stored_types = []

    def per_output_scores(
        self,
        output: GenerateOutputType,
    ) -> torch.Tensor:
        # Retieve probability distribution over the vocabulary for all sequences

        self.batch_size = output.sequences.shape[0] // self.num_return_sequences
        # [len_gen, batch_size*numreturn, vocab_size]

        scores = torch.stack(output.scores).transpose(0, 1)

        probabilities = self.mk_probability(scores)
        vocab_size = probabilities.shape[-1]

        mask = mask_pad_tokens(output.sequences, probabilities, self.pad_token_id)

        prob_types = (probabilities * mask[:, :, None]).sum(1) / mask.sum(-1)[:, None]

        # [batch_size, numreturn, vocab_size]
        prob_types = prob_types.view(
            self.batch_size * self.num_return_sequences, vocab_size
        )

        # [batch_size, numreturn]
        scores = self.projection_function(prob_types).cpu()
        scores = scores.view(self.batch_size, self.num_return_sequences)

        return scores

    def projection_function(self, prob_types: torch.Tensor) -> torch.Tensor:

        pair_wise_information = (
            torch.pow(prob_types[:, None, :], self.alpha)
            / torch.pow(
                self.stored_types_tensor[None, :, :].to(prob_types.get_device()),
                1 - self.alpha,
            )
        ).sum(-1)

        if self.use_soft_projection:
            if self.n_neighbors == -1:
                n_neighbors = pair_wise_information.shape[1] - 1
            else:
                n_neighbors = self.n_neighbors

            scores, _ = torch.topk(
                pair_wise_information, k=n_neighbors, dim=-1, largest=False
            )
            scores = scores.mean(-1)

        else:
            # Return the index of the closest type
            # [batch_size, numreturn, ]
            scores, _ = torch.min(pair_wise_information, 2)

        return scores

    def __format__(self, format_spec):
        return (
            f"{self.__class__.__name__}(alpha={self.alpha}, "
            f"use_soft_projection={self.use_soft_projection},"
            f"n_neighbors={self.n_neighbors},"
            f"temperature={self.temperature}, "
            f"mode={self.mode})"
        )
