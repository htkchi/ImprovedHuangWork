## Import the Libraries:
import numpy as np #work with array/matrix
import matplotlib.pyplot as plt #draw charts
import pandas as pd #ML package

## Import Dataset:
dataset = pd.read_csv('cut_features.csv')
X = dataset.iloc[:, [2,5]].values #Take all values of the row, only column index 3,4
#X = dataset.iloc[:, [2,3,4,5,7,8,9,10]].values #Take all values of the row, only column index 3,4
print(X)
print(len(X))

## Elbow Method:
from sklearn.cluster import KMeans
# wcss = [] #Empty matrix
# for i in range(1, 66):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 66), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

## Train the K-Means model: with K=5, found from above
kmeans = KMeans(int(0.3*len(X)), init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

## Visualizing the clusters:
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 300, c = 'red', label = 'Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 300, c = 'blue', label = 'Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 300, c = 'green', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 300, c = 'cyan', label = 'Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 300, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 500, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of Cut Coefficients')
# plt.xlabel('Mean of Coefficients')
# plt.ylabel('Std of Coefficients')
# plt.legend()
# plt.show()