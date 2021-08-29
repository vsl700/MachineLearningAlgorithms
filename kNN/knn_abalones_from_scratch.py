# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import abalones_dataset_for_knn


X, y = abalones_dataset_for_knn.get_data()

print(X, y, '', sep='\n\n')

new_data_point = np.array([
    0.569552,
    0.446407,
    0.154437,
    1.016849,
    0.439051,
    0.222526,
    0.291208
])

distances = np.linalg.norm(X - new_data_point, axis=1)
print('Distances:', distances)

k = 7
nearest_neighbor_ids = distances.argsort()[:k]  # This method sorts the array and returns the indexes of sorted items
print('Indexes:', nearest_neighbor_ids)

nearest_neighbor_rings = y[nearest_neighbor_ids]
print('Rings:', nearest_neighbor_rings)

# For regression
prediction = nearest_neighbor_rings.mean()
print('\nRegression prediction:', prediction)

# For classification (when we don't have numbers, but strings for example)
# st.mode()[0] - the mode
# st.mode()[1] - the amount of items that are in mode
prediction_class = st.mode(nearest_neighbor_rings)[0]
print('\nClassification prediction:', prediction_class)
