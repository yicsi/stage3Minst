'''
RNN method for Stage 4 IMDb text classification.
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy


class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        length = sum(1 for token_id in sequence if token_id != 0)
        if length == 0:
            length = 1
        return torch.LongTensor(sequence), torch.LongTensor([self.labels[index]]).squeeze(), length


def collate_text_batch(batch):
    sequences, labels, lengths = zip(*batch)
    return (
        torch.stack(sequences),
        torch.LongTensor(labels),
        torch.LongTensor(lengths)
    )


class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 5
    learning_rate = 1e-3
    batch_size = 64

    def __init__(self, mName, mDescription, vocab_size=10000):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = 128
        self.bidirectional = True
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=128,
            padding_idx=0
        )
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=self.hidden_size,
            batch_first=True,
            nonlinearity='tanh',
            bidirectional=self.bidirectional
        )
        self.dropout = nn.Dropout(0.5)
        direction_count = 2 if self.bidirectional else 1
        self.feature_size = self.hidden_size * direction_count
        self.fc = nn.Linear(self.feature_size * 3, 2)
        self.to(self.device)

    def forward(self, x, lengths):
        x = self.embedding(x.long())
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        mask = torch.arange(output.size(1), device=self.device)[None, :] < lengths[:, None]
        masked_output = output * mask.unsqueeze(2)
        mean_pool = masked_output.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)
        max_ready = output.masked_fill(~mask.unsqueeze(2), -1e9)
        max_pool = max_ready.max(dim=1)[0]

        features = torch.cat((final_hidden, mean_pool, max_pool), dim=1)
        y_pred = self.fc(self.dropout(features))
        return y_pred

    def evaluate_dataset(self, loader, loss_function, evaluator_name):
        accuracy_evaluator = Evaluate_Accuracy(evaluator_name, '')
        true_y = []
        pred_y = []
        total_loss = 0.0

        super().train(False)
        with torch.no_grad():
            for batch_X, batch_y, lengths in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                lengths = lengths.to(self.device)
                logits = self.forward(batch_X, lengths)
                loss = loss_function(logits, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                true_y.extend(batch_y.cpu().tolist())
                pred_y.extend(logits.max(1)[1].cpu().tolist())

        accuracy_evaluator.data = {
            'true_y': true_y,
            'pred_y': pred_y
        }
        acc = accuracy_evaluator.evaluate()
        precision = accuracy_evaluator.last_metrics['precision']
        return total_loss / len(loader.dataset), acc, precision

    def train(self, X, y, test_X, test_y):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_function = nn.CrossEntropyLoss()

        train_dataset = TextDataset(X, y)
        test_dataset = TextDataset(test_X, test_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_text_batch
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_text_batch
        )

        epoch_list = []
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        for epoch in range(self.max_epoch):
            super().train(True)
            epoch_loss = 0.0

            for batch_X, batch_y, lengths in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                lengths = lengths.to(self.device)
                y_pred = self.forward(batch_X, lengths)
                train_loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                epoch_loss += train_loss.item() * batch_X.size(0)

            train_loss, train_acc, train_precision = self.evaluate_dataset(
                train_loader,
                loss_function,
                'training evaluator'
            )
            test_loss, test_acc, test_precision = self.evaluate_dataset(
                test_loader,
                loss_function,
                'testing evaluator'
            )

            epoch_list.append(epoch + 1)
            train_loss_list.append(epoch_loss / len(train_dataset))
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            print(
                'Epoch:', epoch + 1,
                'Train Accuracy:', train_acc,
                'Train Precision:', train_precision,
                'Train Loss:', train_loss,
                'Test Accuracy:', test_acc,
                'Test Precision:', test_precision,
                'Test Loss:', test_loss
            )

        self.save_learning_curves(
            epoch_list,
            train_loss_list,
            train_acc_list,
            test_loss_list,
            test_acc_list
        )

    def save_learning_curves(self, epochs, train_losses, train_accs, test_losses, test_accs):
        save_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'result', 'stage_4_result')
        ) + os.sep
        os.makedirs(save_folder, exist_ok=True)

        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('IMDb RNN Train vs Test Loss Curve')
        plt.legend()
        plt.savefig(save_folder + 'rnn_text_classification_loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(epochs, train_accs, label='Train Accuracy')
        plt.plot(epochs, test_accs, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('IMDb RNN Train vs Test Accuracy Curve')
        plt.legend()
        plt.savefig(save_folder + 'rnn_text_classification_accuracy_curve.png')
        plt.close()

    def test(self, X):
        test_dataset = TextDataset(X, np.zeros(len(X), dtype=int))
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_text_batch
        )
        pred_y = []

        super().train(False)
        with torch.no_grad():
            for batch_X, _, lengths in test_loader:
                y_pred = self.forward(batch_X.to(self.device), lengths.to(self.device))
                pred_y.extend(y_pred.max(1)[1].cpu().tolist())

        return pred_y

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y']
        )

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y']
        }
