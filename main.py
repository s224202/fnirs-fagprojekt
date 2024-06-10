from Tools.pipeline_builder import build_pipeline
from Tools.data_loaders import load_individual, load_author_data
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Test the pipeline builder
data = load_individual(0)
labels = data.annotations.to_data_frame()['description']
labels = LabelEncoder().fit_transform(labels)
print(data.annotations)
pipeline1 = build_pipeline('None', 'Spline', 'None', 'None', False)
#print(cross_val_score(MLPClassifier((728,728)), pipeline1.fit_transform(data), labels, cv=3))

pipeline2 = build_pipeline('None', 'None', 'None', 'None', False)
#print(cross_val_score(MLPClassifier((728,728)), pipeline2.fit_transform(data), labels, cv=3))
print(data)
plt.plot(pipeline1.fit_transform(data)[0])
plt.plot(pipeline2.fit_transform(data)[0])
plt.show()

