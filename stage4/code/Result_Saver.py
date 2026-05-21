'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.result import result
import os
import pickle


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        file_path = self.result_destination_folder_path + self.result_destination_file_name + '_' + str(self.fold_count)
        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f)
