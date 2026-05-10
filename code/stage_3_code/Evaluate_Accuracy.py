'''
Concrete Evaluate class for a specific evaluation metrics
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings


class Evaluate_Accuracy(evaluate):
    data = None
    last_metrics = None

    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='The number of unique classes is greater than 50% of the number of samples.*',
                category=UserWarning
            )
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        self.last_metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        if self.data.get('final', False):
            print('******** Final Evaluation Metrics ********')
            print('Final Accuracy:', acc)
            print('Final Precision:', precision)
            print('Final Recall:', recall)
            print('Final F1-score:', f1)

        return acc
