#%%
import pandas as pd
import numpy as np
from scipy import stats
from scikit_posthocs import posthoc_dunn

# read csv files
df = pd.read_csv('Results/results.csv', sep=',', header=None)

dfbase = pd.read_csv('Results/baseline_results.csv', sep=',', header=None)

#%%
# calculate mean and sd of all pipelines

for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mean of pipeline {pipeline}: {round(np.mean(df.iloc[row, 1::2]), 2)}', f'SD of pipeline {pipeline}: {round(np.mean(df.iloc[row, 2::2]),2)}')

# calculate mean and sd of baseline
print(f'Mean of baseline: {round(np.mean(dfbase.iloc[0, 1::2]), 2)}', f'SD of baseline: {round(np.mean(dfbase.iloc[0, 2::2]),2)}')

#%%
none = df.iloc[0, 1::2]
bandpass = df.iloc[1, 1::2]
ica = df.iloc[2, 1::2]
bpca = df.iloc[3, 1::2]
regression = df.iloc[4, 1::2]
tddr = df.iloc[5, 1::2]
wiener = df.iloc[6, 1::2]
spline = df.iloc[7, 1::2]
ica_tddr = df.iloc[8, 1::2]
ica_wiener = df.iloc[9, 1::2]
ica_spline = df.iloc[10, 1::2]
bpca_tddr = df.iloc[11, 1::2]
bpca_wiener = df.iloc[12, 1::2]
bpca_spline = df.iloc[13, 1::2]
regression_tddr = df.iloc[14, 1::2]
regression_wiener = df.iloc[15, 1::2]
regression_spline = df.iloc[16, 1::2]
baseline = dfbase.iloc[0, 1::2]
# are the variances equal?
print(stats.bartlett(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline))

#%%
# shapiro-wilks
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Is {pipeline} normally distributed?', stats.shapiro(df.iloc[row, 1::2]).pvalue > 0.05)

print(f'Is baseline normally distributed?', stats.shapiro(dfbase.iloc[0, 1::2]).pvalue > 0.05)

#%%
# calculate one-way ANOVA and Kruskal-Wallis (without baseline)

print(f'Is there a statistical significant difference between the means?', stats.f_oneway(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline).pvalue < 0.05)
print(f'Is there a statistical significant difference between the means? (no distribution assumptions)', stats.kruskal(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline).pvalue < 0.05)

#%%
# paired t-test for baseline
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Paired t-test for {pipeline} and baseline:', stats.ttest_rel(df.iloc[row, 1::2], baseline).pvalue, stats.ttest_rel(df.iloc[row, 1::2], baseline).pvalue < 0.05)

# %%
