from Tools.pipeline_builder import build_pipeline
from Tools.data_loaders import load_individual
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
# Test the pipeline builder
data = load_individual(0)
labels = data.annotations.to_data_frame()['description']
labels = LabelEncoder().fit_transform(labels)
pipeline = build_pipeline('Wiener', 'None', 'None', 'None')
print(cross_val_score(MLPClassifier((728,728)), pipeline.fit_transform(data), labels, cv=3))