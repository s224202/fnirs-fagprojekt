from Tools.pipeline_builder import build_pipeline
from Tools.data_loaders import load_individual
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# Test the pipeline builder
data = load_individual(0)

# Test the pipeline builder

labels = data.annotations.to_data_frame()['description']
labels = LabelEncoder().fit_transform(labels)
pipeline = build_pipeline('Wiener', 'None', 'None', 'None')
print(cross_val_score(SVC(), pipeline.fit_transform(data), labels, cv=5))