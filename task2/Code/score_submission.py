import pandas as pd
import numpy as np
import sklearn.metrics as metrics

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


def get_score(df_true, df_submission):
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    score = np.mean([task1, task2, task3])
    #print(task1, task2, task3)
    return score, task1, task2, task3


filename_sub = 'Data/Submission_Sven.csv'
df_submission = pd.read_csv(filename_sub)

# generate a baseline based on sample.zip
filename_ground = 'Data/train_labels_sorted.csv'
df_true = pd.read_csv(filename_ground)


for label in TESTS + ['LABEL_Sepsis']:
    # round classification labels
    df_true[label] = np.around(df_true[label].values)

over_all, T1, T2, T3 = get_score(df_true, df_submission)
#print('Score of sample.zip with itself as groundtruth', get_score(df_true, df_submission))

print(f'Score of {filename_sub} with {filename_ground} as groundtruth: {over_all}' )
print(f'Task1: {T1}')
print(f'Task2: {T2}')
print(f'Task3: {T3}')
