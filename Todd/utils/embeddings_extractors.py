from typing import List, Tuple, Dict, Optional, Iterable
from collections import defaultdict
import torch
from torch.utils.data import DataLoader


def extract_vocab_probs(
        reference_distribution,
        output,
) -> Tuple[Dict[Tuple[int, int], List[torch.Tensor]], torch.Tensor]:
    """
    Append new layer embeddings from the output to the provided dictionnary
    """
    new_scores = torch.stack(output.scores).softmax(dim=2).view(-1, output.scores[0].size(1))
    reference_distribution.extend(new_scores)


def extract_batch_embeddings(
        per_layer_embeddings,
        output,
        y: Optional[torch.Tensor] = None,
        layers: Optional[Iterable[int]] = None,
        hidden_states="encoder_hidden_states") -> Tuple[Dict[Tuple[int, int], List[torch.Tensor]], torch.Tensor]:
    """
    Append new layer embeddings from the output to the provided dictionnary
    """
    if layers is None:
        layers = range(len(output[hidden_states]))
    if layers is not None:
        # TODO: make clearer
        N_layers = len(output[hidden_states])
        layers = [l if l >= 0 else N_layers + l for l in layers]

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
