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
        # Initialize centroids
        self.init_centroids(self.n_clusters, input)

        # Number of loop count
        iter_count = 0
        
        # Continue looping as long as loop count is len than max
        # or change between old and new centroid > 0.0001
        while True or iter_count < self.max_iter:
            old_centroids = np.array(self.cluster_centers_)
            self.recenter_centroids(input)
            new_centroids = self.cluster_centers_
            difference = np.abs(np.subtract(new_centroids, old_centroids))
            iter_count += 1

            # Check if the change is smaller or larger than 0.0001
            if (np.any(difference) > 0.0001):
                continue
            else:
                break

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
        # Get a random position, and assign a cluster there.
        pos = randint(1, len(input)-1)
        self.cluster_centers_ = np.array([input[pos]])

        # Continute to get random position and assign as cluster.
        for i in range(1, num_features):
            pos = randint(1, len(input)-1)
            x = np.array([input[pos]])
            self.cluster_centers_ = np.append(self.cluster_centers_, x,axis=0)


    def calculate_distance(self, d_features, c_features) -> int:
        """
            Calculates the Euclidean distance between point A and point B. 
            Recall that a Euclidean distance can be expanded such that D^2 = A^2 + B^2 + ... Z^2. 
        """
        distance_squared = 0 

        for i in range(len(d_features)):
            distance_squared += abs(d_features[i] - c_features[i])**2


        return math.sqrt(distance_squared)

    def recenter_centroids(self, input: np.array) -> None:
        """
            This function recenters the centroid to the average distance of all its datapoints.
            Returns nothing, but updates cluster centers 
        """
        # array to store c index.
        c_array = []

        # for each data point in input
        # calculate the distance from data point to each of the centroid
        # pick out the smallest distance and get the centroid's index
        for data in input:
            distance_array = []
            for i in range(len(self.cluster_centers_)):
                distance_array.append(self.calculate_distance(data, self.cluster_centers_[i]))
            c = np.argmin(distance_array)
            c_array.append(c)
        
        # assign labels as c_array 
        self.labels_= np.array(c_array)

        # then, for each cluster
        # cluster = mean of distance to each data point.
        for i in range(len(self.cluster_centers_)):
            sumDistance = 0
            count = 0
            for j in range(len(c_array)):
                if c_array[j] == i:
                    sumDistance += input[j]
                    count += 1
            self.cluster_centers_[i] = sumDistance/count


 
                    

        