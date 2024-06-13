from re import I
from Tools.pipeline_builder import build_pipeline
from Tools.data_loaders import load_individual, load_author_data, concatenate_data
from Tools.function_wrappers import feature_selection_wrapper
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd

# Test the pipeline builder
# control pipeline
results_list = []
baseline_results = []
pipeline = build_pipeline(systemic='None', motion='None', phys='None', classifier='None', split_epochs=True)

# Chosen pipelines:
# only bandpass
bandpass = build_pipeline(systemic='Band pass', motion='None', phys='None', classifier='None', split_epochs=True)
# only physiological
ica = build_pipeline(systemic='Band pass', motion='None', phys='ICA', classifier='None', split_epochs=True)
bpca = build_pipeline(systemic='Band pass', motion='None', phys='bPCA', classifier='None', split_epochs=True)
regression = build_pipeline(systemic='Band pass', motion='None', phys='Regression', classifier='None', split_epochs=True)
# only motion
tddr = build_pipeline(systemic='Band pass', motion='TDDR', phys='None', classifier='None', split_epochs=True)
Wiener = build_pipeline(systemic='Band pass', motion='Wiener', phys='None', classifier='None', split_epochs=True)
spline = build_pipeline(systemic='Band pass', motion='Spline', phys='None', classifier='None', split_epochs=True)
# mixed ICA
ica_tddr = build_pipeline(systemic='Band pass', motion='TDDR', phys='ICA', classifier='None', split_epochs=True)
ica_wiener = build_pipeline(systemic='Band pass', motion='Wiener', phys='ICA', classifier='None', split_epochs=True)
ica_spline = build_pipeline(systemic='Band pass', motion='Spline', phys='ICA', classifier='None', split_epochs=True)
# mixed bPCA
bpca_tddr = build_pipeline(systemic='Band pass', motion='TDDR', phys='bPCA', classifier='None', split_epochs=True)
bpca_wiener = build_pipeline(systemic='Band pass', motion='Wiener', phys='bPCA', classifier='None', split_epochs=True)
bpca_spline = build_pipeline(systemic='Band pass', motion='Spline', phys='bPCA', classifier='None', split_epochs=True)
# mixed regression
regression_tddr = build_pipeline(systemic='Band pass', motion='TDDR', phys='Regression', classifier='None', split_epochs=True)
regression_wiener = build_pipeline(systemic='Band pass', motion='Wiener', phys='Regression', classifier='None', split_epochs=True)
regression_spline = build_pipeline(systemic='Band pass', motion='Spline', phys='Regression', classifier='None', split_epochs=True)

# Test the feature selection wrapper
model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000)
baselinemodel = DummyClassifier(strategy='most_frequent')
datalist = [load_individual(0), load_individual(1), load_individual(2), load_individual(3), load_individual(4)]
#datalist = [load_author_data(2), load_author_data(3), load_author_data(4)]
labelslist = [datalist[i].annotations.to_data_frame()['description'] for i in range(3)]
pipelines_list = [pipeline,bandpass, ica, bpca, regression, tddr, Wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline]
#data, labels = concatenate_data(datalist, labelslist)
label_encoder = LabelEncoder()
for i in range(3):
    results_list.append([])
    baseline_results.append([])
    labelslist[i] = label_encoder.fit_transform(labelslist[i])
    for j in range(len(pipelines_list)):
        newdata = pipelines_list[j].fit_transform(datalist[i])
        sfs = SequentialFeatureSelector(model, n_features_to_select='auto', cv=3, tol=0.01)
        scores = cross_val_score(sfs, newdata, labelslist[i], cv=3, scoring='accuracy')
        results_list[i].append((scores.mean(), scores.std()))
        scores2 = cross_val_score(baselinemodel,newdata, labelslist[i], cv=3)
        baseline_results[i].append((scores2.mean(), scores2.std()))
        print(f'{(i*17+j)/51*100}% done')

df = pd.DataFrame(results_list)
df.T.to_csv('results.csv')
df = pd.DataFrame(baseline_results)
df.T.to_csv('baseline_results.csv')


#individual 0: (78, 120) labels = (90,)
#individual 1: (4, 108)  labels = (90,)
#individual 2: (28, 96)  labels = (90,)
#individual 3: (35, 156) labels = (90,)
#individual 4: (61, 78)  labels = (90,)