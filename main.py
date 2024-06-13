from Tools.pipeline_builder import build_pipeline
from Tools.data_loaders import load_individual, load_author_data, concatenate_data
from Tools.function_wrappers import feature_selection_wrapper
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

# Test the pipeline builder
pipeline = build_pipeline(systemic='None', motion='None', phys='None', classifier='None', split_epochs=True)

# Test the feature selection wrapper
model = MLPClassifier(hidden_layer_sizes=(100, 100))
baselinemodel = DummyClassifier(strategy='most_frequent')
datalist = [load_individual(0), load_individual(1), load_individual(2), load_individual(3), load_individual(4)]
#datalist = [load_author_data(2), load_author_data(3), load_author_data(4)]
labelslist = [datalist[i].annotations.to_data_frame()['description'] for i in range(5)]

for i in range(5):
    datalist[i] = pipeline.fit_transform(datalist[i])
data, labels = concatenate_data(datalist, labelslist)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
#data = feature_selection_wrapper(data, labels)
scores = cross_val_score(model, data, labels, cv=3)
print(scores)
