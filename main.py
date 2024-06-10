from Tools.pipeline_builder import build_pipeline
from Tools.data_loaders import load_individual, load_author_data
from Tools.function_wrappers import feature_selection_wrapper
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Test the pipeline builder
data = load_individual(0)
labels = data.annotations.to_data_frame()['description']
labels = LabelEncoder().fit_transform(labels)
pipeline = build_pipeline(systemic='Wiener', motion='TDDR', phys='bPCA', classifier='None')
pipeline = pipeline.fit(data)
data = pipeline.transform(data)

# Test the feature selection wrapper
model = MLPClassifier(hidden_layer_sizes=(728, 728))
data = feature_selection_wrapper(data, labels)
scores = cross_val_score(model, data, labels, cv=3)
print(scores)