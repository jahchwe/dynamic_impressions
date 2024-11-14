# in this file we are going to create 2D embeddings of the timeseries data and the final numerical ratings data. 

# the general idea that we are trying to illustrate is that there are groupings in the timeseries data that correspond to groupings in the final numerical ratings data. 

# that is, how you form your timeseries impacts how you form your final numerical ratings.

# this is assessed formally using the RSA analysis, but descriptively it would be good to show clusterings in the timeseries data and then that those clusterings are reflected in the final numerical ratings data.

import pickle
# import mds from sklearn
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import os
import numpy as np

# import squareform
from scipy.spatial.distance import squareform

from hdbscan import HDBSCAN

STUDY_ID = 'first_collect_10_10_2023_test'
VARIANT_ID = 'ratings-euclidean_final-cosine_ts-ctw'
TIMESERIES_PICKLE_PATH = '../rsa_results/%s/%s/pdist_pickles/timeseries_pdist.pkl' % (STUDY_ID, VARIANT_ID)
RATINGS_PICKLE_PATH = '../rsa_results/%s/%s/pdist_pickles/ratings_pdist.pkl' %  (STUDY_ID, VARIANT_ID)
OUTPUT_PATH = '../OUTPUT/ts_ratings_clusterings'

# load the pickles
# NOTE: pickles are dictionaries with keys being the video ID and values being the pdist matrix

# timeseries data
with open(TIMESERIES_PICKLE_PATH, 'rb') as handle:
    timeseries_data = pickle.load(handle)

# ratings data
with open(RATINGS_PICKLE_PATH, 'rb') as handle:
    ratings_data = pickle.load(handle)

# create the MDS object
mds = MDS(n_components=2, dissimilarity='precomputed')
#tsne = TSNE(n_components=2, metric='precomputed', init = 'random')

# iterate thru the videos inside the pickles
for video in timeseries_data.keys():
    ts = squareform(np.asarray(timeseries_data[video]))
    ratings = squareform(np.asarray(ratings_data[video]))

    # cluster the timeseries data using HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=2, metric='precomputed')
    cluster_labels = clusterer.fit_predict(ts)

    # embed timeseries data
    timeseries_embedding = mds.fit_transform(ts)
    # embed ratings data
    ratings_embedding = mds.fit_transform(ratings)
    
    # create a facet plot, plotting the two embeddings side by side

    # create a colorbar for the clusterings
    cmap = plt.cm.rainbow
    norm = colors.Normalize(vmin=np.min(cluster_labels), vmax=np.max(cluster_labels))


    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Timeseries and Ratings Embeddings for %s' % video)
    # scatter with a different color for each point
    ax1.scatter(timeseries_embedding[:,0], timeseries_embedding[:,1], c = cluster_labels, cmap = cmap, norm = norm)
    ax1.set_title('Timeseries Embedding')

    ax2.scatter(ratings_embedding[:,0], ratings_embedding[:,1], c = cluster_labels, cmap = cmap, norm = norm)
    ax2.set_title('Ratings Embedding')

    # save the plot
    plt.savefig(os.path.join(OUTPUT_PATH, '%s.png' % video))