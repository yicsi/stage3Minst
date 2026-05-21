'''
Dataset loader for Stage 4 joke text generation.
'''

import csv
import re
from collections import Counter

from code.base_class.dataset import dataset


class Dataset_Joke_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = 'data'

    def __init__(self, dataset_name, dDescription=None, max_vocab_size=5000):
        dataset.__init__(self, dataset_name, dDescription)
        self.max_vocab_size = max_vocab_size

    def tokenize(self, text):
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
        text = text.replace(' " ', ' "')
        return text

    def load(self):
        print('loading joke generation data...')
        file_path = self.dataset_source_folder_path + self.dataset_source_file_name
        jokes = []

        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                joke = row['Joke'].strip()
                if joke:
                    jokes.append(joke)

        tokenized_jokes = [self.tokenize(joke) for joke in jokes]
        counter = Counter()
        for tokens in tokenized_jokes:
            counter.update(tokens)

        vocabulary = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        for word, _ in counter.most_common(self.max_vocab_size - len(vocabulary)):
            vocabulary[word] = len(vocabulary)

        index_to_word = {index: word for word, index in vocabulary.items()}
        encoded_jokes = []
        for tokens in tokenized_jokes:
            encoded = [vocabulary.get(word, vocabulary['<UNK>']) for word in tokens]
            encoded.append(vocabulary['<EOS>'])
            encoded_jokes.append(encoded)

        print('jokes:', len(encoded_jokes))
        print('vocabulary size:', len(vocabulary))

        return {
            'jokes': jokes,
            'encoded_jokes': encoded_jokes,
            'vocabulary': vocabulary,
            'index_to_word': index_to_word,
            'vocab_size': len(vocabulary)
        }
