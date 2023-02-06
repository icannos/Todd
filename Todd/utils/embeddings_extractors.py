import torch


def extract_vocab_probs(
        reference_distribution,
        output,
) -> None:
    """
    Append new layer embeddings from the output to the provided dictionnary
    """
    new_scores = torch.stack(output.scores).softmax(dim=2).view(-1, output.scores[0].size(1))
    reference_distribution.extend(new_scores)


# We should not offer this function to the user, model manipulation should be done before the call to this library

# def extract_embeddings(
#         model, tokenizer, dataloader: DataLoader, layers: Optional[List[int]] = None
# ) -> Tuple[Dict[Tuple[int, int], List[torch.Tensor]], torch.Tensor]:
#     """
#     Extract the embeddings of the input sequences. Not classified per class.
#     :param layers: List of layers to return. If None, return all layers.
#     :param model: huggingface model
#     :param tokenizer: huggingface tokenizer
#     :param dataloader: dataloader of the input sequences
#     :return: a dictionary with the embeddings of the input sequences
#     """
#     per_layer_embeddings = defaultdict(list)
#
#     with torch.no_grad():
#         for batch in dataloader:
#             # Retrieves hidden states from the model
#             inputs = tokenizer(
#                 batch["source"], padding=True, truncation=True, return_tensors="pt"
#             ).to(model.device)
#             output = model.generate(
#                 **inputs,
#                 return_dict_in_generate=True,
#                 output_hidden_states=True,
#                 output_scores=True,
#             )
#
#             per_layer_embeddings, y = extract_batch_embeddings(
#                 per_layer_embeddings,
#                 output,
#                 layers=layers,
#             )
#
#     return per_layer_embeddings, y
