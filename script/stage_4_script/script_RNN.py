import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split


def optional_int_env(name):
    value = os.environ.get(name)
    if value is None or value.strip() == '':
        return None
    return int(value)


if __name__ == '__main__':
    train_limit = optional_int_env('STAGE4_TRAIN_LIMIT_PER_CLASS')
    test_limit = optional_int_env('STAGE4_TEST_LIMIT_PER_CLASS')
    max_epoch = int(os.environ.get('STAGE4_CLASSIFICATION_EPOCHS', '5'))
    max_vocab_size = int(os.environ.get('STAGE4_MAX_VOCAB_SIZE', '20000'))
    max_sequence_length = int(os.environ.get('STAGE4_MAX_SEQUENCE_LENGTH', '300'))
    remove_stop_words = os.environ.get('STAGE4_REMOVE_STOP_WORDS', '0') == '1'

    dataset = Dataset_Loader(
        'text',
        'IMDb text classification dataset',
        max_vocab_size=max_vocab_size,
        max_sequence_length=max_sequence_length,
        train_limit_per_class=train_limit,
        test_limit_per_class=test_limit,
        remove_stop_words=remove_stop_words
    )
    dataset.dataset_source_folder_path = os.path.join(
        PROJECT_ROOT,
        'data',
        'stage_4_data',
        'stage_4_data',
        'text_classification'
    )

    loaded_data = dataset.load()

    method = Method_RNN(
        'RNN',
        'bidirectional RNN for text classification',
        vocab_size=loaded_data['vocab_size']
    )
    method.max_epoch = max_epoch
    method.data = loaded_data

    result_saver = Result_Saver('saver', '')
    result_saver.result_destination_folder_path = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result') + os.sep
    result_saver.result_destination_file_name = 'RNN_text_classification_prediction_result'

    evaluate = Evaluate_Accuracy('accuracy', '')

    print('************ Start Stage 4 RNN Text Classification ************')
    result = method.run()
    result_saver.data = result
    result_saver.save()

    evaluate.data = {
        'true_y': result['true_y'],
        'pred_y': result['pred_y'],
        'final': True
    }
    accuracy = evaluate.evaluate()
    print('Final Accuracy:', accuracy)
    print('************ Finish Stage 4 RNN Text Classification ************')
