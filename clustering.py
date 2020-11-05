"""

IDE: PyCharm
Project: pdf-explorer
Author: Robin
Filename: clustering.py
Date: 05.11.2020

"""
import numpy as np
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


@st.cache
def do_tsne(vectors):
    X = np.array(vectors)

    # visualize clusters
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(X)
    return tsne_results


@st.cache
def do_clustering(words, vectors, eps, min_samples):
    tsne_results = do_tsne(vectors)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(tsne_results)

    noise_counter, cluster_counter = 0, 0
    labels = clustering.labels_
    clusters = [[] for label in labels]
    num_clusters = set()
    for i in range(len(labels)):
        word_id = i
        label_id = labels[i]
        # ignore noise = -1
        if label_id >= 0:
            clusters[label_id].append(words[word_id])
            cluster_counter += 1
            num_clusters.add(label_id)
        else:
            noise_counter += 1
    return clusters, num_clusters, noise_counter, cluster_counter, tsne_results, labels
