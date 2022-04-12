import numpy as np
import pandas as pd

sample = 'sample.csv'
pred_st1 = 'pred_st1.csv'
pred_st2 = 'pred_st2.csv'
pred_st3 = 'pred_st3.csv'

df_sample = pd.read_csv(sample)
df_st1 = pd.read_csv(pred_st1)
df_st2 = pd.read_csv(pred_st2)
df_st3 = pd.read_csv(pred_st3)

df_sub = df_st1.join([df_st2, df_st3])

assert df_sub.shape == df_sample.shape
assert df_sub.columns == df_sub.columns

df_sub.to_csv('Submision.csv', index=False, float_format='%.3f', compression='zip')
