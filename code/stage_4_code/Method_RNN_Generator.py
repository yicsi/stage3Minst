'''
RNN method for Stage 4 joke text generation.
'''

import os
import re

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from code.base_class.method import method


class JokeDataset(Dataset):
    def __init__(self, encoded_jokes):
        self.samples = [torch.LongTensor(joke) for joke in encoded_jokes if len(joke) > 1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence = self.samples[index]
        return sequence[:-1], sequence[1:]


def collate_jokes(batch):
    inputs, targets = zip(*batch)
    input_batch = pad_sequence(inputs, batch_first=True, padding_value=0)
    target_batch = pad_sequence(targets, batch_first=True, padding_value=0)
    return input_batch, target_batch


class Method_RNN_Generator(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 32

    def __init__(self, mName, mDescription, vocab_size=5000):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.rnn = nn.RNN(128, 160, batch_first=True)
        self.fc = nn.Linear(160, vocab_size)
        self.to(self.device)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x.long())
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def train_model(self):
        dataset = JokeDataset(self.data['encoded_jokes'])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_jokes)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=0)
        losses = []

        for epoch in range(self.max_epoch):
            super().train(True)
            total_loss = 0.0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                logits, _ = self.forward(batch_X)
                loss = loss_function(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

            avg_loss = total_loss / len(dataset)
            losses.append(avg_loss)
            print('Epoch:', epoch + 1, 'Generation Loss:', avg_loss)

        self.save_learning_curve(losses)
        return losses

    def save_learning_curve(self, losses):
        save_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'result', 'stage_4_result')
        ) + os.sep
        os.makedirs(save_folder, exist_ok=True)
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Joke RNN Generation Training Loss')
        plt.legend()
        plt.savefig(save_folder + 'rnn_text_generation_loss_curve.png')
        plt.close()

    def tokenize_seed(self, text):
        text = text.lower()
        text = re.sub(r'([,.!?;:()"\'])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().split()

    def detokenize(self, tokens):
        text = ' '.join(tokens)
        for mark in ['.', ',', '!', '?', ';', ':', ')']:
            text = text.replace(' ' + mark, mark)
        text = text.replace('( ', '(')
        text = text.replace(" ' ", "'")
        return text

    def generate(self, seed_words, max_new_words=35, temperature=0.8):
        vocabulary = self.data['vocabulary']
        index_to_word = self.data['index_to_word']
        tokens = self.tokenize_seed(seed_words)
        if not tokens:
            tokens = ['what', 'did', 'the']

        generated = tokens[:]
        input_ids = [vocabulary.get(word, vocabulary['<UNK>']) for word in tokens]
        hidden = None

        super().train(False)
        with torch.no_grad():
            for token_id in input_ids[:-1]:
                input_tensor = torch.LongTensor([[token_id]]).to(self.device)
                _, hidden = self.forward(input_tensor, hidden)

            current_id = input_ids[-1]
            for _ in range(max_new_words):
                input_tensor = torch.LongTensor([[current_id]]).to(self.device)
                logits, hidden = self.forward(input_tensor, hidden)
                logits = logits[:, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1).item()

                if next_id == vocabulary['<EOS>']:
                    break
                next_word = index_to_word.get(next_id, '<UNK>')
                if next_word not in {'<PAD>', '<UNK>'}:
                    generated.append(next_word)
                current_id = next_id

        return self.detokenize(generated)

    def run(self):
        print('method running...')
        print('--start generation training...')
        losses = self.train_model()
        return {'losses': losses}
