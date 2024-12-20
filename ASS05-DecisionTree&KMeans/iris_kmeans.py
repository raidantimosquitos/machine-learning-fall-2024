# Iris K-Means segmentation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
def load_dataset():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=['sl', 'sw', 'pl', 'pw'])
    y = pd.Series(iris.target)
    return X, y

# Compute the euclidean distance between two points
def euclidean_dist(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

# Implementing E - Expectation step (assign data points to the nearest cluster center) 
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X.iloc[idx]
        
        for i in range(k):
            dis = euclidean_dist(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters
        
# Implementing the M - Maximization step (updates cluster centers based on the mean 
# of the assigned points in K-means clustering)
def update_clusters(clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
            
            clusters[i]['points'] = []
    return clusters

# Predict the cluster which data point belongs to
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(euclidean_dist(X.iloc[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred 

if __name__ == '__main__':
    # print the iris dataset samples
    X, y = load_dataset()
    fig = plt.figure(0)
    plt.grid(True)
    plt.scatter(X.iloc[:,0], X.iloc[:,1])
    plt.show()

    # initialize the random centroids for the K-means algorithm
    k = 3 # 3-classes
    clusters = {}

    for idx in range(k):
        center = [(np.mean(X.iloc[:,i]) + 1 - 2*np.random.rand()) for i in range(X.shape[1])]
        points = []
        cluster = {'center' : center, 'points' : []}
        clusters[idx] = cluster

    # Plot and print initial centroids for each cluster
    print('Initial cluster centroid coordinates')
    plt.scatter(X.iloc[:,0],X.iloc[:,1])
    plt.grid(True)
    for i in clusters:
        center = clusters[i]['center']
        print(f'Centroid {i}: {center}')
        plt.scatter(center[0],center[1],marker = '*',c = 'red')
    plt.show()

    # Assign, update and predict the cluster center
    clusters = assign_clusters(X, clusters)
    clusters = update_clusters(clusters)
    pred = pred_cluster(X, clusters)

    
    # Plot the final cluster centroids and compute accuracy
    plt.scatter(X.iloc[:,0], X.iloc[:,1], c = pred)
    print()
    print('Final cluster centroid coordinates')
    for i in clusters:
        center = clusters[i]['center']
        print(f'Centroid {i}: {center}')
        plt.scatter(center[0], center[1], marker = '^', c = 'red')
    plt.grid(True)
    plt.show()

    # Print cross tab predictions
    print()
    print("Confusion matrix: col_0 = predicted_classes, row_1 = target_classes") 
    print(pd.crosstab(y, pred))