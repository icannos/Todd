from typing import Union

import torch
from transformers.generation import (
    BeamSampleDecoderOnlyOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSampleEncoderDecoderOutput,
)

# Generate type output:

GenerateOutputType = Union[
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
]


def extract_log_probability_distributions(
    output: GenerateOutputType,
    normalize: bool = False,
) -> torch.Tensor:
    """
    output: GenerateOutputType = Union[
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
        BeamSampleDecoderOnlyOutput,
        BeamSampleEncoderDecoderOutput,
    ]
    Return the step by step probability distributions of the output sequences.
    Shape: (num_return_sequences, sequence_length, vocab_size)
    """

    scores = output.scores
    sequences = output.sequences
    beam_indices = output.beam_indices if hasattr(output, "beam_indices") else None

    # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
    # to a beam search approach were the first (and only) beam is always selected
    if beam_indices is None:
        beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
        beam_indices = beam_indices.expand(-1, len(scores))

    scores = torch.stack(scores).transpose(0, 1)

    # renormalize probabilities
    if normalize:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)

    # 4. cut beam_indices to longest beam length
    beam_indices_mask = beam_indices < 0
    # max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()

    beam_indices[beam_indices_mask] = 0

    # Some weird hackiness:
    max_beam_length = scores.shape[1]
    beam_indices = beam_indices[:, :max_beam_length]

    # 5. Set indices of beams that finished early to 0
    # such indices will be masked correctly afterwards

    # 8. Compute scores
    beam_indices = beam_indices[..., None].expand(-1, -1, scores.shape[-1])

    transition_scores = torch.gather(scores, dim=0, index=beam_indices)

    return transition_scores


def extract_decoder_hidden_states(
    output: GenerateOutputType,
    hidden_layer_idx=-1,
):

    scores = output.scores
    sequences = output.sequences
    beam_indices = output.beam_indices if hasattr(output, "beam_indices") else None

    if isinstance(output, BeamSearchDecoderOnlyOutput) or isinstance(
        output, BeamSampleDecoderOnlyOutput
    ):
        decoder_hidden_states = output.hidden_states
    elif isinstance(output, BeamSearchEncoderDecoderOutput) or isinstance(
        output, BeamSampleEncoderDecoderOutput
    ):
        decoder_hidden_states = output.decoder_hidden_states
    else:
        raise ValueError("Unknown output type")

    # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
    # to a beam search approach were the first (and only) beam is always selected
    if beam_indices is None:
        beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
        beam_indices = beam_indices.expand(-1, len(scores))

    # handling of the target length and preparing the masking for tokens
    # outside of that length
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]
    beam_indices[beam_indices_mask] = 0

    # seqlen = sequences.shape[1] - 1

    # creating the output hidden_states representation in format:
    # [bsz * beam_width ; seqlen ; featdim]
    decoder_hidden_states = torch.stack(
        [
            decoder_hidden_states[i][hidden_layer_idx][:, 0, :].index_select(
                dim=0, index=beam_indices[:, i]  # reordering using the beam_indices
            )
            for i in range(len(scores))
        ]
    ).transpose(0, 1)

    # setting to 0 the hidden_states were it doesn't make sense to have an output
    decoder_hidden_states[beam_indices_mask] = 0

    return decoder_hidden_states


def extract_hidden_state(output, chosen_state, hidden_layer_idx=-1):
    if chosen_state == "encoder_hidden_states":
        return output["encoder_hidden_states"][hidden_layer_idx]
    return extract_decoder_hidden_states(output, hidden_layer_idx=hidden_layer_idx)
