import torch
from typing import List, Tuple
from Todd.query_based_scorers.scoring_functions import ScoringFunction, MaxSoftmaxProbability, CrossEntropyLoss, SoftmaxEntropy, RenyiDivergence, RenyiDivergenceWithReference
from Todd.query_based_scorers.scoring_aggregators import ScoringAggregator, MeanAggregator, MaxAggregator, MaskedMaxAggregator, MaskedMeanAggregator, MaskedSumAggregator, SumAggregator

from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        x = self.sentences[idx]
        return x


class QueryBasedScorer:
    def __init__(self,
                 model=None,
                 tokenizer=None,
                 batch_size: int = 32,
                 prefix: str = None,
                 suffix: str = None,
                 scoring_functions: List[ScoringFunction] = None,
                 scoring_aggregator: List[ScoringAggregator] = None,
                 device: str = None,
                 *args,
                 **kwargs):

        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device) if model is not None else None
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefix = prefix
        self.suffix = suffix
        self.reference_probs = []
        self.concat_output = False

        if scoring_functions:
            self.scoring_functions = scoring_functions
        else:
            self.scoring_functions = [CrossEntropyLoss(),
                                      SoftmaxEntropy(),
                                      MaxSoftmaxProbability(),
                                      RenyiDivergence(),
                                      RenyiDivergenceWithReference(alpha=0.5),
                                      RenyiDivergenceWithReference(alpha=0.9),
                                      RenyiDivergenceWithReference(alpha=2.0),
                                      RenyiDivergenceWithReference(alpha=3.0),
                                      RenyiDivergenceWithReference(alpha=5.0),
                                      RenyiDivergenceWithReference(alpha=10.0),
                                      RenyiDivergenceWithReference(alpha=20.0)]
        if scoring_aggregator:
            self.scoring_aggregator = scoring_aggregator
        else:
            self.scoring_aggregator = [MeanAggregator(),
                                       MaxAggregator(),
                                       MaskedMaxAggregator(),
                                       MaskedMeanAggregator(),
                                       # MaskedSumAggregator(),
                                       # SumAggregator()
                                       ]

    def score_tokens(self, logits, labels, mask=None):
        scores = []
        for scoring_function in self.scoring_functions:
            token_scores = scoring_function(logits, labels)
            for aggregator in self.scoring_aggregator:
                if mask.sum() == 0:
                    agg_score = 0
                else:
                    agg_score = aggregator(token_scores, mask).nan_to_num(nan=0., posinf=100, neginf=-100).item()
                scores.append({f"{scoring_function.name}_{aggregator.name}": agg_score})
        return scores

    def score_sentences(self, sentences: List[str], model=None, tokenizer=None):
        model = model if self.model is None else self.model
        tokenizer = tokenizer if self.tokenizer is None else self.tokenizer
        return [self.score_sentence(sentence, model, tokenizer) for sentence in sentences]

    def score_sentence(self, sentence: str, model=None, tokenizer=None):
        logits, labels, masked_index = self.prepare_sentence(sentence, model, tokenizer)
        scores = self.score_tokens(logits, labels.to(model.device), mask=masked_index)
        return scores

    def prepare_sentence(self, sentence: str, model=None, tokenizer=None):
        raise NotImplementedError

    def accumulate(self, sentences: List[str], model=None, tokenizer=None):
        model = model if self.model is None else self.model
        tokenizer = tokenizer if self.tokenizer is None else self.tokenizer

        for sentence in sentences:
            logits, labels, masked_index = self.prepare_sentence(sentence, model, tokenizer)
            self.reference_probs.append(logits[masked_index].softmax(dim=1).mean(dim=0).cpu())

    def fit(self):
        self.reference_probs = torch.stack(self.reference_probs).mean(dim=0)
        for scoring_func in self.scoring_functions:
            if isinstance(scoring_func, RenyiDivergenceWithReference):
                scoring_func.reference = self.reference_probs

    def return_masked_input(self, sentence: str, tokenizer):
        raise NotImplementedError


class TrainableQueryBasedScorer(QueryBasedScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_output = True
        self.training_data = []
        self.learning_rate = kwargs.get("learning_rate", 1e-5)
        self.num_train_epochs = kwargs.get("num_train_epochs", 10)
        self.device= kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def prepare_sentence(self, sentence: Tuple[str, str], model=None, tokenizer=None):
        raise NotImplementedError

    def accumulate(self, sentences: List[str], model=None, tokenizer=None) -> None:
        super().accumulate(sentences, model, tokenizer)

        for sentence in sentences:
            if self.prefix is not None:
                sentence = (sentence[0].replace(self.prefix, ""), sentence[1])
            if self.suffix is not None:
                sentence = (sentence[0].replace(self.suffix, ""), sentence[1])
            self.training_data.append(sentence)


    def fit(self):
        super().fit()
        dataset = CustomDataset(self.training_data)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        def collate_batch(batch):
            sentence_list, label_list, token_type_list = [], [], []
            for sent in batch:
                # Process sentence, repetition is turned off for now
                # TODO: Position IDs restart after each sequence
                input_ids, labels, token_type_ids = self.return_masked_input(sent, self.tokenizer, repetitions=1)
                sentence_list.append(input_ids.squeeze())
                label_list.append(labels.squeeze())
                if token_type_ids is not None:
                    token_type_list.append(token_type_ids.squeeze())

            x = torch.nn.utils.rnn.pad_sequence(sentence_list, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.model.device)
            t = None
            if len(token_type_list) > 0:
                t = torch.nn.utils.rnn.pad_sequence(token_type_list, batch_first=True, padding_value=1).to(self.model.device)
            y = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.model.device)
            del sentence_list, label_list
            y[~(x == self.tokenizer.mask_token_id)] = -100
            return x, y, t


        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch
        )
        # Torch training loop
        self.model.train()

        def train_loop(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            for batch, (X, y, t) in enumerate(dataloader):
                # Compute prediction and loss
                pred = model(X, token_type_ids=t).logits
                loss = loss_fn(pred.view(-1, pred.shape[-1]), y.view(-1))

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        def test_loop(dataloader, model, loss_fn):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            test_loss, correct = 0, 0

            with torch.no_grad():
                for X, y in dataloader:
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        for t in range(self.num_train_epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, self.model, loss_fn, optimizer)
            # test_loop(train_dataloader, self.model, loss_fn)
        print("Done!")

        # Clear the accumulated embeddings
        self.accumulated_embeddings = []
        self.labels = []

        del self.training_data