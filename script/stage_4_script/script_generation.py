import os
import pickle
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from code.stage_4_code.Dataset_Joke_Loader import Dataset_Joke_Loader
from code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator


if __name__ == '__main__':
    max_epoch = int(os.environ.get('STAGE4_GENERATION_EPOCHS', '20'))
    max_vocab_size = int(os.environ.get('STAGE4_GENERATION_MAX_VOCAB_SIZE', '5000'))

    dataset = Dataset_Joke_Loader(
        'joke generation',
        'short joke text generation dataset',
        max_vocab_size=max_vocab_size
    )
    dataset.dataset_source_folder_path = os.path.join(
        PROJECT_ROOT,
        'data',
        'stage_4_data',
        'stage_4_data',
        'text_generation'
    ) + os.sep
    loaded_data = dataset.load()

    method = Method_RNN_Generator(
        'RNN generator',
        'RNN for joke text generation',
        vocab_size=loaded_data['vocab_size']
    )
    method.max_epoch = max_epoch
    method.data = loaded_data

    print('************ Start Stage 4 RNN Text Generation ************')
    result = method.run()

    seeds = [
        'what did the',
        'why did the',
        'knock knock who',
        'machine learning is'
    ]
    generated_examples = {}
    for seed in seeds:
        generated_examples[seed] = method.generate(seed, max_new_words=35, temperature=0.8)

    result_folder = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result')
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, 'RNN_text_generation_examples.txt'), 'w', encoding='utf-8') as file:
        for seed, text in generated_examples.items():
            file.write('Seed: ' + seed + '\n')
            file.write(text + '\n\n')

    with open(os.path.join(result_folder, 'RNN_text_generation_result_None'), 'wb') as file:
        pickle.dump(
            {
                'losses': result['losses'],
                'generated_examples': generated_examples,
                'vocab_size': loaded_data['vocab_size']
            },
            file
        )

    print('************ Generated Examples ************')
    for seed, text in generated_examples.items():
        print('Seed:', seed)
        print(text)
    print('************ Finish Stage 4 RNN Text Generation ************')
