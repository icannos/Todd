from typing import Dict, Tuple, List, Optional
from transformers import BertConfig, BertModel

import torch
from torch.utils.data import DataLoader, Dataset

from transformers.modeling_outputs import ModelOutput

from Todd.basescorers import HiddenStateBasedScorers
from Todd.utils.output_processing import extract_hidden_state


class LogitClassifier(torch.nn.Module):
    def __init__(self, hidden_size, num_labels=2, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = BertConfig(hidden_size=hidden_size, num_hidden_layers=6)
        self.base_model = BertModel(config=config).encoder.to(self.device)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels).to(self.device)

    def forward(self, x):
        x = self.base_model(x).last_hidden_state
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        assert len(embeddings) == len(labels)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.Tensor(self.embeddings[idx])
        y = self.labels[idx]
        return x, y


class ClassifierScorer(HiddenStateBasedScorers):
    def __init__(self, layers: List[int] = (-1,), **kwargs):
        super().__init__(kwargs)
        self.layers = -1    # set(layers)
        self.accumulated_embeddings = []
        self.labels = []

        hidden_size = kwargs.get("hidden_size", 768)
        device = kwargs.get("device", None)
        self.num_train_epochs = kwargs.get("num_train_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 8)
        self.learning_rate = kwargs.get("learning_rate", 3e-5)
        self.model = LogitClassifier(hidden_size=hidden_size, device=device)

    def compute_scores_benchmark(
        self, output: ModelOutput
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Compute the Mahalanobis distance of the first sequence returned for each input.
        :param output: output of the model
        :return: (*,) tensor of Mahalanobis distances
        """
        hidden_state = extract_hidden_state(
            output,
            self.chosen_state,
            hidden_layer_idx=self.layers
        )

        scores = self.model(hidden_state).softmax(dim=1)[:, 1].tolist()
        scores = {
            f"score": scores
        }
        return scores

    def accumulate(self, output: ModelOutput, y: Optional[List[int]] = None) -> None:
        hidden_state = extract_hidden_state(
            output,
            self.chosen_state,
            hidden_layer_idx=self.layers
        ).tolist()
        self.accumulated_embeddings.extend(hidden_state)
        # If labels is empty, we are in the first batch
        self.labels.extend(y*(len(hidden_state)))

    def fit(self):
        dataset = CustomDataset(self.accumulated_embeddings, self.labels)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        def collate_batch(batch):
            label_list, embed_list, = [], []
            for (_embed, _label) in batch:
                label_list.append(_label)
                embed_list.append(_embed)
            x = torch.nn.utils.rnn.pad_sequence(embed_list, batch_first=True, padding_value=0).to(self.model.device)
            y = torch.Tensor(label_list).long().to(self.model.device)
            return x, y

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
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)

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
