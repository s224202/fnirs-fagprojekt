#%%
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests, fdrcorrection

# For CUH  (DoC import resultsriget.csv or healthy results_DoC_healthy.csv)
# read csv files
df = pd.read_csv('Results/resultsriget.csv', sep=',', header=None)

#%%
# calculate mean and sd of all pipelines
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mean of pipeline {pipeline}: {round(np.mean(df.iloc[row, 1::6]), 3)}', f'SD of pipeline {pipeline}: {round(np.mean(df.iloc[row, 2::6]), 3)}')

# calculate mean and sd of recall of all pipelines
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mean of recall {pipeline}: {round(np.mean(df.iloc[row, 3::6]), 3)}', f'SD of recall {pipeline}: {round(np.mean(df.iloc[row, 4::6]), 3)}')

# calculate f1-score of all pipelines
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mean of f1-score {pipeline}: {round(np.mean(df.iloc[row, 5::6]), 3)}', f'SD of f1-score {pipeline}: {round(np.mean(df.iloc[row, 6::6]), 3)}')

#%%
none = df.iloc[0, 1::6]
bandpass = df.iloc[1, 1::6]
ica = df.iloc[2, 1::6]
bpca = df.iloc[3, 1::6]
regression = df.iloc[4, 1::6]
tddr = df.iloc[5, 1::6]
wiener = df.iloc[6, 1::6]
spline = df.iloc[7, 1::6]
ica_tddr = df.iloc[8, 1::6]
ica_wiener = df.iloc[9, 1::6]
ica_spline = df.iloc[10, 1::6]
bpca_tddr = df.iloc[11, 1::6]
bpca_wiener = df.iloc[12, 1::6]
bpca_spline = df.iloc[13, 1::6]
regression_tddr = df.iloc[14, 1::6]
regression_wiener = df.iloc[15, 1::6]
regression_spline = df.iloc[16, 1::6]
baseline = 0.5
#%%
# are the variances equal?
print(f'Are the pipelines variances equal?', stats.bartlett(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue > 0.05, stats.bartlett(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue)

#%%
# shapiro-wilks
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Is {pipeline} normally distributed?', stats.shapiro(df.iloc[row, 1::6]).pvalue > 0.05)

#%%
# calculate one-way ANOVA and Kruskal-Wallis (without baseline)

print(f'Is there a statistical significant difference between the means?', stats.f_oneway(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue < 0.05, stats.f_oneway(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue)
print(f'Is there a statistical significant difference between the means? (no distribution assumptions)', stats.kruskal(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue < 0.05, stats.kruskal(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue)

#%%
# mann whitney U test for pipelines vs baseline
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mann-Whitney test for {pipeline} and baseline:', stats.mannwhitneyu(df.iloc[row, 1::6], [0.5 for _ in range(len(df.iloc[row, 1::6]))], nan_policy='omit').pvalue, stats.mannwhitneyu(df.iloc[row, 1::6], [0.5 for _ in range(len(df.iloc[row,1::6]))], nan_policy='omit').pvalue < 0.05)

print(stats.mannwhitneyu(df.iloc[3, 1::6], [0.5 for _ in range(len(df.iloc[3, 1::6]))], nan_policy='omit').pvalue)
# adjusted p-values with fdr correctionn (benjamini-yekutieli)
pvalues = [stats.mannwhitneyu(df.iloc[row, 1::6], [0.5 for _ in range(len(df.iloc[row,1::6]))], nan_policy='omit').pvalue for row in range(17)]
print(fdrcorrection(pvalues, alpha=0.05, method='n', is_sorted=False))
#%%
# For CUH DoC patients
Patient1 = df.iloc[:, 1]
Patient2 = df.iloc[:, 7]
Patient3 = df.iloc[:, 13]
Patient4 = df.iloc[:, 19]
Patient5 = df.iloc[:, 25]
Patient6 = df.iloc[:, 31]
Patient7 = df.iloc[:, 37]
Patient8 = df.iloc[:, 43]
Patient9 = df.iloc[:, 49]
Patient10 = df.iloc[:, 55]
Patient11 = df.iloc[:, 61]
Patient12 = df.iloc[:, 67]
Patient13 = df.iloc[:, 73]
Patient14 = df.iloc[:, 79]
Patient15 = df.iloc[:, 85]
Patient16 = df.iloc[:, 91]
Patient17 = df.iloc[:, 97]
Patient18 = df.iloc[:, 103]
Patient19 = df.iloc[:, 109]
Patient20 = df.iloc[:, 115]
Patient21 = df.iloc[:, 121]
Patient22 = df.iloc[:, 127]
Patient23 = df.iloc[:, 133]
Patient24 = df.iloc[:, 139]
Patient25 = df.iloc[:, 145]
Patient26 = df.iloc[:, 151]
Patient27 = df.iloc[:, 157]
Patient28 = df.iloc[:, 163]
Patient29 = df.iloc[:, 169]
Patient30 = df.iloc[:, 175]
Patient31 = df.iloc[:, 181]
Patient32 = df.iloc[:, 187]
Patient33 = df.iloc[:, 193]
Patient34 = df.iloc[:, 199]
Patient35 = df.iloc[:, 205]
Patient36 = df.iloc[:, 211]

#%%

for column, patient in enumerate(["Column", "Patient 1", "Patient 2", "Patient 3","Patient 4","Patient 5","Patient 6","Patient 7","Patient 8","Patient 9","Patient 10","Patient 11","Patient 12","Patient 13","Patient 14","Patient 15","Patient 16","Patient 17","Patient 18","Patient 19","Patient 20","Patient 21","Patient 22","Patient 23","Patient 24","Patient 25","Patient 26","Patient 27","Patient 28","Patient 29","Patient 30","Patient 31","Patient 32","Patient 33","Patient 34","Patient 35","Patient 36"]):
    print(f'Mean of {patient}: {round(np.mean(df.iloc[:, 1::6]), 3)}', f'SD of {patient}: {round(np.mean(df.iloc[:, 2::6]), 3)}')
#%%

# are the variances equal?
print(f'Are the patients variances equal?', stats.bartlett(Patient1, Patient2, Patient3, Patient4, Patient5, Patient6, Patient7, Patient8, Patient9, Patient10, Patient11, Patient12, Patient13, Patient14, Patient15, Patient16, Patient17, Patient18, Patient19, Patient20, Patient21, Patient22, Patient23, Patient24, Patient25, Patient26, Patient27, Patient28, Patient29, Patient30, Patient31, Patient32, Patient33, Patient34, Patient35, Patient36, nan_policy='omit').pvalue > 0.05, stats.bartlett(Patient1, Patient2, Patient3, Patient4, Patient5, Patient6, Patient7, Patient8, Patient9, Patient10, Patient11, Patient12, Patient13, Patient14, Patient15, Patient16, Patient17, Patient18, Patient19, Patient20, Patient21, Patient22, Patient23, Patient24, Patient25, Patient26, Patient27, Patient28, Patient29, Patient30, Patient31, Patient32, Patient33, Patient34, Patient35, Patient36, nan_policy='omit').pvalue)

#%%

# shapiro-wilks
for column, patient in enumerate(["Column", "Patient 1", "Patient 2", "Patient 3","Patient 4","Patient 5","Patient 6","Patient 7","Patient 8","Patient 9","Patient 10","Patient 11","Patient 12","Patient 13","Patient 14","Patient 15","Patient 16","Patient 17","Patient 18","Patient 19","Patient 20","Patient 21","Patient 22","Patient 23","Patient 24","Patient 25","Patient 26","Patient 27","Patient 28","Patient 29","Patient 30","Patient 31","Patient 32","Patient 33","Patient 34","Patient 35","Patient 36"]):
    print(f'Is {patient} normally distributed?', stats.shapiro(df.iloc[:, 1::6]).pvalue > 0.05)

#%%
# mann-whitney u for patients vs baseline
for column, patient in enumerate(["Column", "Patient 1", "Patient 2", "Patient 3","Patient 4","Patient 5","Patient 6","Patient 7","Patient 8","Patient 9","Patient 10","Patient 11","Patient 12","Patient 13","Patient 14","Patient 15","Patient 16","Patient 17","Patient 18","Patient 19","Patient 20","Patient 21","Patient 22","Patient 23","Patient 24","Patient 25","Patient 26","Patient 27","Patient 28","Patient 29","Patient 30","Patient 31","Patient 32","Patient 33","Patient 34","Patient 35","Patient 36"]):
    print(f'Mann-Whitney test for {patient} and baseline:', (stats.mannwhitneyu(df.T.iloc[column, 1:126:6], [0.5 for _ in range(len(df.T.iloc[column, 1:126:6]))], nan_policy='omit').pvalue, stats.mannwhitneyu(df.T.iloc[column, 1:126:6], [0.5 for _ in range(len(df.T.iloc[column, 1:126:6]))], nan_policy='omit').pvalue < 0.05))
#%%

# adjusted p-values with fdr correctionn (benjamini-yekutieli)
pvalues = [stats.mannwhitneyu(df.T.iloc[column, 1:126:6], [0.5 for _ in range(len(df.T.iloc[column, 1:126:6]))], nan_policy='omit').pvalue for column in range(37)]
print(fdrcorrection(pvalues, alpha=0.05, method='n', is_sorted=False))

#%%
# For CUH DoC healthy controls
Participant1 = df.iloc[:, 1]
Participant2 = df.iloc[:, 7]
Participant3 = df.iloc[:, 13]
Participant4 = df.iloc[:, 19]
Participant5 = df.iloc[:, 25]

#%%
for column, participant in enumerate(["Column", "Participant 1", "Participant 2", "Participant 3","Participant 4","Participant 5"]):
    print(f'Mean of {participant}: {round(np.mean(df.iloc[:, 1::6]), 3)}', f'SD of {participant}: {round(np.mean(df.iloc[:, 2::6]), 3)}')

#%%
# are the variances equal?
print(f'Are the participants variances equal?', stats.bartlett(Participant1, Participant2, Participant3, Participant4, Participant5, nan_policy='omit').pvalue > 0.05, stats.bartlett(Participant1, Participant2, Participant3, Participant4, Participant5, nan_policy='omit').pvalue)

#%%
# shapiro-wilks
for column, participant in enumerate(["Column", "Participant 1", "Participant 2", "Participant 3","Participant 4","Participant 5"]):
    print(f'Is {participant} normally distributed?', stats.shapiro(df.iloc[:, 1::6]).pvalue > 0.05)

#%%
# mann-whitney u for participants vs baseline
for column, participant in enumerate(["Column", "Participant 1", "Participant 2", "Participant 3","Participant 4","Participant 5"]):
    print(f'Mann-Whitney test for {participant} and baseline:', (stats.mannwhitneyu(df.T.iloc[column, 1:26:6], [0.5 for _ in range(len(df.T.iloc[column, 1:26:6]))], nan_policy='omit').pvalue, stats.mannwhitneyu(df.T.iloc[column, 1:26:6], [0.5 for _ in range(len(df.T.iloc[column, 1:26:6]))], nan_policy='omit').pvalue < 0.05))

# adjusted p-values with fdr correctionn (benjamini-yekutieli)
pvalues = [stats.mannwhitneyu(df.T.iloc[column, 1:26:6], [0.5 for _ in range(len(df.T.iloc[column, 1:26:6]))], nan_policy='omit').pvalue for column in range(6)]
print(fdrcorrection(pvalues, alpha=0.05, method='n', is_sorted=False))












#%%
# For Rob Lukes
# read csv files
df = pd.read_csv('Results/resultsrobFinal.csv', sep=',', header=None)

#%%
# calculate mean and sd of all pipelines

for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mean of pipeline {pipeline}: {round(np.mean(df.iloc[row, 1::2]), 3)}', f'SD of pipeline {pipeline}: {round(np.mean(df.iloc[row, 2::2]), 3)}')

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
baseline = 0.3333
#%%
# are the variances equal?
print(f'Are the pipelines variances equal?', stats.bartlett(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue > 0.05, stats.bartlett(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue)

#%%
# shapiro-wilks
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Is {pipeline} normally distributed?', stats.shapiro(df.iloc[row, 1::2]).pvalue > 0.05)

#%%
# calculate one-way ANOVA and Kruskal-Wallis (without baseline)

print(f'Is there a statistical significant difference between the means?', stats.f_oneway(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue < 0.05, stats.f_oneway(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue)
print(f'Is there a statistical significant difference between the means? (no distribution assumptions)', stats.kruskal(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue < 0.05, stats.kruskal(none, bandpass, ica, bpca, regression, tddr, wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline, nan_policy='omit').pvalue)


#%%
# mann-whitney test for pipelines vs baseline
for row, pipeline in enumerate(["None", "Band-pass", "ICA", "bPCA", "sc regression", "TDDR", "Wiener", "Spline", "ICA-TDDR", "ICA-Wiener", "ICA-Spline", "bPCA-TDDR", "bPCA-Wiener", "bPCA-Spline", "sc regression-TDDR", "sc regression-Wiener", "sc regression-Spline"]):
    print(f'Mann-Whitney for {pipeline} and baseline:', stats.mannwhitneyu(df.iloc[row, 1::2], [0.333 for _ in range(len(df.iloc[row,1::2]))]).pvalue, stats.mannwhitneyu(df.iloc[row, 1::2], [0.333 for _ in range(len(df.iloc[row,1::2]))]).pvalue < 0.05)

#%%

# adjusted p-values with fdr correctionn (benjamini-yekutieli)
pvalues = [stats.mannwhitneyu(df.iloc[row, 1::2], [0.333 for _ in range(len(df.iloc[row,1::2]))]).pvalue for row in range(17)]
print(fdrcorrection(pvalues, alpha=0.05, method='n', is_sorted=False))

#%%
# for participants
Participant1 = df.iloc[:, 1]
Participant2 = df.iloc[:, 3]
Participant3 = df.iloc[:, 5]
Participant4 = df.iloc[:, 7]
Participant5 = df.iloc[:, 9]

#%%
for column, participant in enumerate(["Column", "Participant 1", "Participant 2", "Participant 3","Participant 4","Participant 5"]):
    print(f'Mean of {participant}: {round(np.mean(df.iloc[:, 1::2]), 3)}', f'SD of {participant}: {round(np.mean(df.iloc[:, 2::2]), 3)}')

