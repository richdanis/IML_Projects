import numpy as np
import pandas as pd

sample = 'sample.csv'
pred_st1 = 'pred_st1.csv'
pred_st2 = 'pred_st2.csv'
pred_st3 = 'pred_st3.csv'

df_sample = pd.read_csv(sample)
df_st1 = pd.read_csv(pred_st1)
#df_st2 = pd.read_csv(pred_st2)
#df_st3 = pd.read_csv(pred_st3)

TEST1 = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
TEST2 = ['LABEL_Sepsis']
TEST3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

df_sample_1 = df_sample[TEST1]
df_sample_2 = df_sample[TEST2]
df_sample_3 = df_sample[TEST3]

df_sub = df_st1.join([df_sample_2, df_sample_3])

assert df_sub.shape == df_sample.shape
assert df_sub.columns == df_sub.columns

df_sub.to_csv('Submision.csv', index=False, float_format='%.3f', compression='zip')
