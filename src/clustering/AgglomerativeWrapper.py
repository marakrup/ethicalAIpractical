import configparser
import pathlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go


class AgglomorativeWrapper:

    def __init__(self, config, data):
        self.config = config
        self.n_clusters = int(self.config['NoClusters'])
        self.labels = AgglomerativeClustering(n_clusters = self.n_clusters).fit_predict(data)
        self.extract_representations(data)

    def extract_representations(self, X, mode='medoid'):
        """
        Calculates the representats for each cluster using the model stored at self.model.
        This can occur in two ways:
        1. centroid: The mean of all points in the cluster. This is not a real user, and is not representative if
        the cluster is curved.
        2. medoids: The cluster point which is closest to the centroid.
        :return: List of representants
        """
        if mode == 'centroid':
            self.representants = self.centroids(X)
        elif mode == 'medoid':
            self.representants = self.medoids(X)
        else:
            raise Exception("Not a valid mode")
        self.repr_indeces = [np.nonzero(np.all(X == repr, axis=1))[0][0] for repr in self.representants]

    def centroids(self, X):
        centers = np.zeros(shape=(self.n_clusters, len(X[0]))) # no clusters, and dimensionality
        for label in range(self.n_clusters):
            centers[label] = np.mean(X[self.labels == label], axis=0)
        return centers

    def medoids(self, X):
        """
        Caluclates the medoid, meaning the point of the cluster which is clostest to its center
        """
        centers = np.zeros(shape=(self.n_clusters, len(X[0])))
        for label in range(self.n_clusters):
            centroid = np.mean(X[self.labels == label], axis=0) # get center
            dists = cosine_distances(centroid.reshape(1, -1), X[self.labels == label]) # calculate all cosine dist to center
            centers[label] = X[self.labels == label][np.argmin(dists[0])] # take minimum distance
        return centers

    def predict(self, user):
        locations = self.representants
        dists = cosine_distances(user.reshape(1, -1), locations)
        return np.argmin(dists[0])

    def visualize(self, data, repr, points=None):
        labels = self.labels
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2],
                                   mode='markers',
                                   marker=dict(size=1),
                                   text=labels,
                                   hoverinfo="text+name",
                                   marker_color=labels, opacity=1, name="Historic Users"))
        fig.add_trace(go.Scatter3d(x=repr[:, 0], y=repr[:, 1], z=repr[:, 2],
                                   mode='markers',
                                   marker=dict(
                                       size=2),
                                   text=list(range(len(repr))),
                                   hoverinfo="text+name",
                                   marker_color=list(range(len(repr))), name="Exemplars"))

        for (label, point) in points:
            fig.add_trace(
                go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]],
                             marker_symbol=['diamond'],
                             marker=dict(
                                 size=3),
                             hoverinfo="name",
                             mode='markers', name=label)
            )
        fig.update_layout(height=800)  # todo configure
        self.figure = fig

    def get_exemplar_of_cluster(self, index):
        if index > len(self.representants):
            raise ValueError
        return self.representants[index], self.repr_indeces[index]
