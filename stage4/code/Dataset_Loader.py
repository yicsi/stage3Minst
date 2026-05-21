'''
Dataset loader for Stage 4 text classification.
'''

import os
import random
import re
from collections import Counter

from code.base_class.dataset import dataset


STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the',
    'to', 'was', 'were', 'will', 'with'
}


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    def __init__(
        self,
        dataset_name,
        dDescription=None,
        max_vocab_size=10000,
        max_sequence_length=200,
        train_limit_per_class=None,
        test_limit_per_class=None,
        remove_stop_words=False,
        random_seed=170
    ):
        dataset.__init__(self, dataset_name, dDescription)
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.train_limit_per_class = train_limit_per_class
        self.test_limit_per_class = test_limit_per_class
        self.remove_stop_words = remove_stop_words
        self.random_seed = random_seed

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        if self.remove_stop_words:
            words = [word for word in words if word not in STOP_WORDS]
        return words

    def read_split(self, split_name, limit_per_class):
        split_path = os.path.join(self.dataset_source_folder_path, split_name)
        texts = []
        labels = []
        rng = random.Random(self.random_seed + (0 if split_name == 'train' else 1))

        for label_name, label in [('neg', 0), ('pos', 1)]:
            folder = os.path.join(split_path, label_name)
            file_names = sorted(name for name in os.listdir(folder) if name.endswith('.txt'))
            if limit_per_class is not None:
                rng.shuffle(file_names)
                file_names = file_names[:limit_per_class]

            for file_name in file_names:
                file_path = os.path.join(folder, file_name)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    texts.append(file.read())
                labels.append(label)

        return texts, labels

    def build_vocabulary(self, tokenized_texts):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        vocabulary = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in counter.most_common(self.max_vocab_size - len(vocabulary)):
            vocabulary[word] = len(vocabulary)
        return vocabulary

    def encode(self, tokens, vocabulary):
        if len(tokens) > self.max_sequence_length:
            head_length = self.max_sequence_length // 2
            tail_length = self.max_sequence_length - head_length
            tokens = tokens[:head_length] + tokens[-tail_length:]

        ids = [vocabulary.get(word, vocabulary['<UNK>']) for word in tokens]
        while len(ids) < self.max_sequence_length:
            ids.append(vocabulary['<PAD>'])
        return ids

    def load(self):
        print('loading data...')
        train_texts, train_y = self.read_split('train', self.train_limit_per_class)
        test_texts, test_y = self.read_split('test', self.test_limit_per_class)

        train_tokens = [self.clean_text(text) for text in train_texts]
        test_tokens = [self.clean_text(text) for text in test_texts]
        vocabulary = self.build_vocabulary(train_tokens)

        train_X = [self.encode(tokens, vocabulary) for tokens in train_tokens]
        test_X = [self.encode(tokens, vocabulary) for tokens in test_tokens]

        print('training samples:', len(train_X))
        print('testing samples:', len(test_X))
        print('vocabulary size:', len(vocabulary))

        return {
            'train': {'X': train_X, 'y': train_y},
            'test': {'X': test_X, 'y': test_y},
            'vocabulary': vocabulary,
            'vocab_size': len(vocabulary)
        }
