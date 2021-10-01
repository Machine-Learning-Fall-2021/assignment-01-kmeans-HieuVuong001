import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average
import math
from random import randint

from numpy.lib.index_tricks import c_


class KMeans():
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter=max_iter
        self.random_state=random_state
        self.cluster_centers_ = None # numpy array # see create_random_centroids for more info
        self.labels_ = None # predictions # numpy array of size len(input)

    def fit(self, input: np.ndarray) -> np.array: 
        """
            Fitting a model means to train the model on some data using your specific algorithm. 
            Typically, you also provide a set of a labels along with your input.
            However, this is an unsupervised algorithm so we don't have y (or labels) to consider! 
                If you're not convinced, look up any supervised learning algorithm on sklearn: https://scikit-learn.org/stable/supervised_learning.html
                If you can explain the difference between the fit function of this unsupervised algorithm and any other supervised algorith, you get 5 extra credit points towards this assignment. 
            This function will simply return the cluster centers, but it will also update your cluster centers and predictions.
        """
        # YOUR CODE HERE
        self.init_centroids(self.n_clusters, input)
        # print("Original cluster is")
        # old_centroids = np.array(self.cluster_centers_)
        # print(old_centroids)
        # print("_____________")
        # self.recenter_centroids(input)
        # print("After recenter:")
        # new_centroids = self.cluster_centers_
        # print(new_centroids)
        # difference = np.subtract(old_centroids, new_centroids)

        condition = True
        iter_count = 0
        while condition or iter_count < self.max_iter:
            # print("Original cluster is")
            old_centroids = np.array(self.cluster_centers_)
            # print(old_centroids)
            # print("_____________")
            self.recenter_centroids(input)
            # print("After recenter:")
            new_centroids = self.cluster_centers_
            difference = np.abs(np.subtract(new_centroids, old_centroids))
            # print("Difference is ")
            # print(difference)
            # print(new_centroids)
            iter_count += 1
            if (np.any(difference) > 0.0001):
                continue
            else:
                break
            # condition = np.all(difference > 0.0001)
        
        # print(self.labels_)
        return self.cluster_centers_
    
    
    def init_centroids(self, num_features: int, input) -> np.array:
        """
            To initialize the classifier, you will create random starting points for your centroids. 
            You will have n_cluster (N) amounts of centroids and the dimension of your centroids depends on the shape of your data.
            For example, your data may have 100 rows and 5 features (M). Therefore, your centroids will need 5 dimensions as well. 
            This function will return nothing, but it will initialize your cluster centers. 
            cluster_centers_ is an attribute that you will update. 
            It has a specific shape of (N, M) where 
                N = n_clusters 
                M = number of features
        """
        # YOUR CODE HERE 
        pos = randint(1, len(input)-1)
        self.cluster_centers_ = np.array([input[pos]])
        for i in range(1, num_features):
            pos = randint(1, len(input)-1)
            x = np.array([input[pos]])
            self.cluster_centers_ = np.append(self.cluster_centers_, x,axis=0)


    def calculate_distance(self, d_features, c_features) -> int:
        """
            Calculates the Euclidean distance between point A and point B. 
            Recall that a Euclidean distance can be expanded such that D^2 = A^2 + B^2 + ... Z^2. 
        """
        # YOUR CODE HERE
        distance_squared = 0 
        for i in range(len(d_features)):
            distance_squared += abs(d_features[i] - c_features[i])
        
        distance_squared = distance_squared**2

        return math.sqrt(distance_squared)

    def recenter_centroids(self, input: np.array) -> None:
        """
            This function recenters the centroid to the average distance of all its datapoints.
            Returns nothing, but updates cluster centers 
        """

        # YOUR CODE HERE 
        c_array = []
        for data in input:
            distanceArray = []
            for i in range(len(self.cluster_centers_)):
                distanceArray.append(self.calculate_distance(data, self.cluster_centers_[i]))
            c = np.argmin(distanceArray)
            c_array.append(c)
        
        self.labels_= np.array(c_array)
        # print(c_array)

        for i in range(len(self.cluster_centers_)):
            sumDistance = 0
            count = 0
            for j in range(len(c_array)):
                if c_array[j] == i:
                    sumDistance += input[j]
                    count += 1
            # print("COUNT IS ")
            # print(count)
            self.cluster_centers_[i] = sumDistance/count


 
                    

        