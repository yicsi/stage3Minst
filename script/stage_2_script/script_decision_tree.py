from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_MLP
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np


# ---- MLP script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(1)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    data_obj = Dataset_Loader('mnist', '')
    data_obj.dataset_source_folder_path = '../../data/'

    method_obj = Method_MLP('mlp', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    
    
    
    