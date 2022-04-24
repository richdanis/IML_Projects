import numpy as np
import pandas as pd
import sklearn.metrics as metrics

TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TEST2 = ['LABEL_Sepsis']
TEST3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

#get score
def get_score(df_true, df_submission):
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task1, task2, task3, = 0,0,0
    task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TEST1])
    print(f'Score in ST1: {task1}')
    task2 = metrics.roc_auc_score(df_true[TEST2[0]], df_submission[TEST2[0]])
    print(f'Score in ST2: {task2}')
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in TEST3])
    print(f'Score in ST3: {task3}')
    print(f'Score: {np.mean([task1, task2, task3])}')
    return
