# Todd: Text Out of Distribution Detection

Todd is a library designed to provide anomaly scorer and the derived filters for text
generation. It supports three levels of anomaly detection: input level, generated sequence 
level and token level. The input level anomaly detection is used for OOD detection. Whereas 
the generated sequence level aims at selecting the best candidates in a beam and the token
level aims at detecting potentially wrong tokens in the generated sequence.

It revolves around two concepts: scorers and filters. The scorers are used to score the input, token or ouput 
and filters leverage these scores to produce masks to decide what to keep or reject.

For benchmarking purpose we tend to only use scorers to dump the scores and compte statistics. 
In practice though, we would want to use filters to select the best candidates.


## Installation

### From github

```bash
git clone git@github.com:icannos/Todd.git
cd Todd
pip install -e .
```

### From pip

```bash
pip install todd
```


## Examples

Different examples are provided in the `examples` folder.

### OOD Detection with Mahalanobis Distance
 
```python

# Extract features from the reference set
ref_embeddings, _ = extract_embeddings(model, tokenizer, in_val_loader, layers=[6])

# Initialize the Mahalanobis detector
maha_detector = MahalanobisScorer(layers=[6])
# Fit the detector with the reference set
maha_detector.fit(ref_embeddings)

with torch.no_grad():
    for batch in loader:
        inputs = tokenizer(
            batch["source"], padding=True, truncation=True, return_tensors="pt"
        )
        output = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
        )

        print(maha_detector(output)) 
        # Output the scores for each input in the batch
        # True means the input is In-Distribution (ie to be kept) and False means OOD
```


## Todo:

- [ ] Add reorderers to reorder the beam search candidates
- [ ] Split filters and anomaly scorers (WIP, maxime)
- [x] Add power means and cosines ood scores

## API