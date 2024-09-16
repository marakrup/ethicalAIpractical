import configparser
import pathlib
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go


class KMeansWrapper:
    def __init__(self, X):
        self.get_cluster_config()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto").fit(X)
        self.labels = self.model.predict(X)
        self.extract_representations(X)  # the "user story" of each cluster

    def get_cluster_config(self):
        """
        Gets parameters from config file and sets them as class attributes
        """
        config = configparser.ConfigParser()
        file_path = pathlib.Path(__file__).parent.parent.parent / 'config.ini'
        config.read(file_path)
        self.config = config[config['DEFAULT']['Dimensionality']]
        self.n_clusters = int(self.config['NoClusters'])

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
            self.representants = self.centroids()
        elif mode == 'medoid':
            self.representants = self.medoids(X)
        else:
            raise Exception("Not a valid mode")

        self.repr_indeces = [np.nonzero(np.all(X == repr, axis=1))[0][0] for repr in self.representants]

    def centroids(self):
        return self.model.cluster_centers_

    def medoids(self, data_points):
        """
        Caluclates the medoid, meaning the point of the cluster which is clostest to its center
        """
        centers = self.model.cluster_centers_
        repr = np.zeros(shape=(self.n_clusters, len(data_points[0])))
        for label, center in enumerate(centers):
            dists = cosine_distances(center.reshape(1, -1), data_points[self.labels == label])
            repr[label] = data_points[self.labels == label][np.argmin(dists[0])]
        return repr

    def predict(self, user):
        return self.model.predict(user[np.newaxis, ...])[0]

    def visualize(self, data, repr, points):
        labels = self.model.labels_
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
        fig.update_layout(height=600)
        self.figure = fig

    def get_exemplar_of_cluster(self, id):
        """ Returns both the value of the exemplar of a cluster as well as the index it has in the historic users. """
        if id > len(self.representants):
            raise ValueError
        return self.representants[id], self.repr_indeces[id]
