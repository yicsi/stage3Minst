'''
Concrete SettingModule class: train/test split for Stage 3
'''

from code.base_class.setting import setting


class Setting_Train_Test_Split(setting):
    dataset = None
    method = None
    result = None
    evaluate = None

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        self.method.data = loaded_data

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        self.evaluate.data['final'] = True
        score = self.evaluate.evaluate()

        return score, 0
