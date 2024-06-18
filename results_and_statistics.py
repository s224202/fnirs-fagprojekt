#%%
import pandas as pd
import numpy as np

# read csv file
df = pd.read_csv('Results/results.csv', sep=',', header=None)

#%%
# calculate mean and sd of all pipelines

for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "sc Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mean of pipeline {pipeline}: {round(np.mean(df.iloc[row, 1::2]), 2)}', f'SD of pipeline {pipeline}: {round(np.mean(df.iloc[row, 2::2]),2)}')

#%%



