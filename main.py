from re import I
from Tools.pipeline_builder import build_pipeline
from Tools.data_loaders import *
from Tools.function_wrappers import feature_selection_wrapper
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler

# Test the pipeline builder
# control pipeline
r = 42
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
model = MLPClassifier(hidden_layer_sizes=(10), max_iter=10000, random_state=r, solver='adam', activation='relu')
#model = SVC(random_state=r, kernel='linear')
baselinemodel = DummyClassifier(strategy='most_frequent')
#datalist = [load_individual(0), load_individual(1), load_individual(2), load_individual(3), load_individual(4)]
#datalist = [load_author_data(2), load_author_data(3), load_author_data(4)]
#datalist = [load_CUH_data(1, 'Healthy'), load_CUH_data(2, 'Healthy'), load_CUH_data(3, 'Healthy'), load_CUH_data(4, 'Healthy'), 
            #load_CUH_data(5, 'Healthy'), 
            #load_CUH_data(5, 'Healthy'), load_CUH_data(7, 'Healthy')]
datalist = [load_CUH_data(1, 'DoC'), load_CUH_data(2, 'DoC'), load_CUH_data(5, 'DoC'), 
            #load_CUH_data(3, 'DoC'), load_CUH_data(6, 'DoC'),
            load_CUH_data(4, 'DoC'), load_CUH_data(7, 'DoC')]
#data, labels = concatenate_data(datalist, labelslist)
pipelines_list = [pipeline, bandpass, ica, bpca, regression, tddr, Wiener, spline, ica_tddr, ica_wiener, ica_spline, bpca_tddr, bpca_wiener, bpca_spline, regression_tddr, regression_wiener, regression_spline]
labelslist = [datalist[i].annotations.to_data_frame()['description'] for i in range(len(datalist))]
label_encoder = LabelEncoder()
persons = []
for i in range(1):
    persons.append([])
    results_list.append([])
    baseline_results.append([])
    labelslist[i] = label_encoder.fit_transform(labelslist[i])
    # accs = []
    # recs = []
    # f1s = []
    # accstds = []
    # recstds = []
    # f1stds = []
    for j in range(17):
        newdata = pipelines_list[j].fit_transform(datalist[i])
        # sfs = SequentialFeatureSelector(model, n_features_to_select='auto', cv=10, tol=0.01, n_jobs=-1)
        # sfs.fit(newdata, labelslist[i])
        # sfs.transform(newdata)
        scoring = {'rec': 'recall_macro', 'f1': 'f1_macro', 'acc': 'accuracy'}
        smallrecs = []
        smallf1s = []
        smallaccs = []

        for train, test in StratifiedKFold(n_splits=10, shuffle=True, random_state=r).split(newdata, labelslist[i]):
                model.fit(newdata[train], labelslist[i][train])
                smallrecs.append(recall_score(labelslist[i][test], model.predict(newdata[test]), average='binary', pos_label=1))
                smallf1s.append(f1_score(labelslist[i][test], model.predict(newdata[test]), average='binary', pos_label=1))
                smallaccs.append(accuracy_score(labelslist[i][test], model.predict(newdata[test])))
        # accs.append(np.mean(smallaccs))
        # recs.append(np.mean(smallrecs))
        # f1s.append(np.mean(smallf1s))
        # accstds.append(np.std(smallaccs))
        # recstds.append(np.std(smallrecs))
        # f1stds.append(np.std(smallf1s))

        #results_list[i].append((scores.mean(), scores.std()))
        #scores2 = cross_val_score(baselinemodel,newdata, labelslist[i], cv=3)
        #baseline_results[i].append((scores2.mean(), scores2.std()))
        print(f'{(i*17+j)/(len(datalist)*17)*100}% done')
        persons[i].append(([np.mean(smallaccs), np.std(smallaccs), np.mean(smallrecs), np.std(smallrecs), np.mean(smallf1s), np.std(smallf1s)]))
        #open('results.txt', 'a').write(sfs.get_support().__str__())

# df = pd.DataFrame(results_list)
# df.T.to_csv('results.csv')
# df = pd.DataFrame(baseline_results)
# df.T.to_csv('baseline_results.csv')

# save the acc, f1 and rec results to a csv
df = pd.DataFrame(persons)
df.T.to_csv('results.csv')

# testpipe1 = build_pipeline(systemic='Band pass', motion='None', phys='None', classifier='None', split_epochs=False) 
# testpipe2 = build_pipeline(systemic='Band pass', motion='Spline', phys='None', classifier='None', split_epochs=False)
# testdata1 = load_CUH_data(5, 'DoC')
# testdata2 = load_CUH_data(5, 'DoC')
# testdata1 = testpipe1.fit_transform(testdata1)
# testdata2 = testpipe2.fit_transform(testdata2)

# # plot the results
# testdata1.plot(n_channels=10, scalings='auto', duration=20, show=False, title='No filter')
# testdata2.plot(n_channels=10, scalings='auto', duration=20, show=False, title='ICA')

# plt.show()