import gensim.downloader as api
from gensim.models import KeyedVectors
import torch
import pandas as pd
import tslearn
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

import umap
import umap.plot
from umap import validation

from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.patheffects import SimpleLineShadow

from sklearn import metrics
from sklearn.cluster import DBSCAN

import os

# COMPONENTS
# 0. Preprocess the data using standard NLP techniques
# 1. Model the topics of the words for each video. We need discrete functions for modeling topics using clustering on word embeddings and
# using traditional topic modeling techniques. Make these separate functions. 
# Based on the topic results, let's model the timeseries of the topics.

# create output folders 

def create_output_folders(dataset_id):
    # create directory for modeling 
    if not os.path.exists('modeling'):
        os.makedirs('modeling')

    if not os.path.exists('modeling/%s' % dataset_id):
        os.makedirs('modeling/%s' % dataset_id)

    if not os.path.exists('modeling/%s/derivatives' % dataset_id):
        os.makedirs('modeling/%s/derivatives' % dataset_id)

    if not os.path.exists('modeling/%s/output' % dataset_id):
        os.makedirs('modeling/%s/output' % dataset_id)
    
    DERIVATIVES_PATH = 'modeling/%s/derivatives' % dataset_id
    OUTPUT_PATH = 'modeling/%s/output' % dataset_id

    return DERIVATIVES_PATH, OUTPUT_PATH

##############################
# dimensionality reduction and clustering functions
##############################

def reduce_PCA(embeddings, components=0.95, DERIVATIVES_PATH=None):
    pca = PCA(n_components=components)
    pca.fit(embeddings)

    # plot the PCA components
    PCA_NUM_DIMS = len(pca.components_)
    plt.bar(x=np.arange(1, PCA_NUM_DIMS+1, 1), height=pca.explained_variance_ratio_.cumsum())   
    plt.savefig(os.path.join(DERIVATIVES_PATH, 'pca_components.png'))
    plt.clf()
    return pca.transform(embeddings)

def reduce_UMAP(embeddings, components=2, metric='cosine', DERIVATIVES_PATH=None):
    reducer = umap.UMAP(n_components=components, metric=metric)
    embedding = reducer.fit_transform(embeddings)

    # plot the UMAP components
    plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1)
    plt.savefig(os.path.join(DERIVATIVES_PATH, 'umap_components.png'))
    plt.clf()
    return embedding

def kmeans_clustering(embeddings, cluster_range = (2, 15), DERIVATIVES_PATH=None):
    silhouettes = []
    for i in range(cluster_range[0], cluster_range[1]):
        kmeans = KMeans(n_clusters=i, init='k-means++').fit(embeddings)
        silhouette = metrics.silhouette_score(embeddings, kmeans.labels_, metric='cosine')
        silhouettes.append(silhouette)
    
    best_n_clusters = np.argmax(silhouettes) + cluster_range[0]

    # plot the silhouette scores
    plt.plot(range(cluster_range[0], cluster_range[1]), silhouettes)
    plt.savefig(os.path.join(DERIVATIVES_PATH, 'silhouette_scores.png'))
    plt.clf()

    kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++').fit(embeddings)
    return kmeans.labels_

##############################
# timeseries analysis functions
##############################

def interpolate(p1, p2, points):
    """Interpolates between two points in 3D space.

    Args:
    p1: The first point.
    p2: The second point.
    t: The interpolation factor.

    Returns:
    The interpolated point.
    """
    interpolated = np.vstack([(1 - t) * p1 + t * p2 for t in np.linspace(0,1, points)])
    return interpolated

def interpolate_timeseries(ts, num_points=500):
    """Interpolates between two points in XD space.
    timeseries is a 2D np.array with columns of timepoints and embedding dimensions
    """

    # sort array by timepoint (first column)

    # assert that the first timepoint is 0
    print(ts[0, 0])
    assert ts[0, 0] == 0

    # rescale trajectory to a standardized number of timepoints
    ts[:, 0] = np.rint((ts[:, 0] / max(ts[:, 0])) * num_points)

    # note: that some entries in the timeseries are by chance so close to the end of the video
    # that they are being rounded to 500, meaning that there is nothing to interpolate between that
    # and the final impression. If there is such an impression, manually set its timepoint to 499
    if ts[-2, 0] == num_points:
        ts[-2, 0] = num_points - 1
    
    out_array = []
    for i in range(ts.shape[0]-1):
        num_points_interpolate = int(ts[i+1, 0] - ts[i, 0])
        print(num_points_interpolate)
        interpolated = interpolate(ts[i,1:], ts[i+1,1:], num_points_interpolate)
        # add global time timepoints
        timepoints = np.arange(ts[i, 0], ts[i+1, 0], 1)
        interpolated = np.insert(interpolated, 0, timepoints, axis=1)
        out_array.extend(interpolated)

    out_array = np.asarray(out_array)
    assert out_array.shape[0] == num_points
    
    return out_array

def plot_timeseries(ts, labels, top_words = None, OUTPUT_PATH=None, xlim = (-20, 20), ylim = (-20, 20)):
    """Plots a 2D trajectory in 3D space.

    Args:
    ts: a np.array with dimensions of subjects x timepoints x embedding dimensions
    labels: clustering labels for visualization, 1D list of numbers indicating category membership, equivalent to the number of subjects
    top_words: a dataframe containing most frequent words, columns of label and embedding dimensions
    """
    # this is the same as the number of timepoints in the trajectory
    NUM_ITER = ts.shape[1]  

    print(ts.shape)
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set(xlim=xlim, xlabel='X')
    ax.set(ylim=ylim, ylabel='Y')

    # plot the top words before animating
    if top_words is not None:
        for it, row in top_words.iterrows():
            plt.text(row['dim1'], row['dim2'], row['preprocessed'], fontsize=8, color='black', alpha=0.5)   


    # create colors equal to the number of unique labels
    colors = sb.color_palette("hls", len(np.unique(labels)))

    lines = [ax.plot([], [], color = colors[labels[i]], alpha=0.2)[0] for i in range(ts.shape[0])]
    points = [ax.plot([], [], 'o', color = colors[labels[i]], alpha=0.6)[0] for i in range(ts.shape[0])]

    def animate(num, data, lines):
        for it, line in enumerate(lines):
            line_data = data[it]
            line.set_data(line_data[:num, 0:2].T)

            point = points[it]
            point.set_data(line_data[num, 0:2].T)

    ani = matplotlib.animation.FuncAnimation(
        fig, animate, NUM_ITER, fargs=(ts, lines), interval=10)

    writervideo = matplotlib.animation.FFMpegWriter(fps=30)
    ani.save(os.path.join(OUTPUT_PATH, 'timeseries.mp4'), writer=writervideo)
    plt.clf()



##############################
# main function
##############################

def model(dataset_id):
    # load data that contains embeddings

    DERIVATIVES_PATH, OUTPUT_PATH = create_output_folders(dataset_id)

    timeseries_cleaned = pd.read_csv('embeddings/%s/timeseries_w_embeddings.csv' % dataset_id, index_col=None)
    final_impressions_cleaned = pd.read_csv('embeddings/%s/final_impressions_w_embeddings.csv' % dataset_id, index_col = None)

    all_data = pd.concat([timeseries_cleaned.reset_index(drop=True), final_impressions_cleaned.reset_index(drop=True)], axis=0)
    
    for it, df in tqdm(all_data.groupby('video')):
        print("Analyzing video %s" % it)
        print(df.shape)
        # make video output folder
        if not os.path.exists(os.path.join(OUTPUT_PATH, it)):
            os.makedirs(os.path.join(OUTPUT_PATH, it))
        if not os.path.exists(os.path.join(DERIVATIVES_PATH, it)):
            os.makedirs(os.path.join(DERIVATIVES_PATH, it))
        
        # calculate sorted word frequencies
        word_frequencies = df.groupby('preprocessed').size().reset_index(name='counts').sort_values(by='counts', ascending=False)
        # write word frequencies to file
        word_frequencies.to_csv(os.path.join(DERIVATIVES_PATH, it, 'word_frequencies.csv'))

        # identify the moments eliciting most responding
        sb.histplot(data=df[df.timepoint != 0], x="timepoint", kde=True, bins = 50)
        plt.savefig(os.path.join(DERIVATIVES_PATH, it, 'timepoint_histogram.png'))
        plt.clf()
                        
        # get unique words
        unique_words_df = df.drop_duplicates(subset=['preprocessed'])

        # reduce dimensionality
        embeddings_reduced = reduce_UMAP(unique_words_df.iloc[:, -300:], components=2, DERIVATIVES_PATH=os.path.join(DERIVATIVES_PATH, it))

        # cluster embeddings
        labels = kmeans_clustering(embeddings_reduced, cluster_range=(2, 15), DERIVATIVES_PATH=os.path.join(DERIVATIVES_PATH, it))

        # plot the reduced embeddings with the kmeans labels
        plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], c=labels, s=0.1)
        plt.savefig(os.path.join(DERIVATIVES_PATH, it, 'umap_components_kmeans.png'))
        plt.clf()
    
        # merge the reduced embeddings onto the original dataframe
        embeddings_reduced_df = pd.DataFrame(embeddings_reduced, columns=['dim1', 'dim2'])
        embeddings_reduced_df['preprocessed'] = unique_words_df.preprocessed.values
        df_w_reduced = df[['subjID', 'video', 'timepoint', 'preprocessed']].reset_index(drop=True).merge(embeddings_reduced_df, on='preprocessed', how='left')

        df_w_reduced.to_csv(os.path.join(DERIVATIVES_PATH, it, 'reduced_no_timepoint_avg.csv'))

        # average together the embeddings for identical timepoints
        df_w_reduced = df_w_reduced.groupby(['subjID', 'timepoint']).agg({'dim1': 'mean', 'dim2': 'mean', 'preprocessed': lambda x: ', '.join(x)}).reset_index()
        df_w_reduced.to_csv(os.path.join(DERIVATIVES_PATH, it, 'reduced_w_timepoint_avg.csv'))
        
        # add kmeans labels to the dataframe
        # output word counts per cluster

        print(df_w_reduced.head())
        print(df_w_reduced.columns)

        # interpolate between timepoints
        interpolated_timeseries_collect = pd.DataFrame()
        NUM_POINTS = 500
        for it_s, s in df_w_reduced.groupby('subjID'):
            print("Interpolating subject %s" % it_s)
            print(s.columns)
            interpolated = interpolate_timeseries(s[['timepoint', 'dim1', 'dim2']].sort_values(by='timepoint').to_numpy(), num_points=NUM_POINTS)
            interpolate_df = pd.DataFrame(interpolated, columns = ['timepoint', 'dim1', 'dim2'])
            interpolate_df['subjID'] = it_s
            interpolated_timeseries_collect = pd.concat([interpolated_timeseries_collect, interpolate_df], axis=0)
        
        interpolated_timeseries_collect.to_csv(os.path.join(DERIVATIVES_PATH, it, 'interpolated_timeseries.csv'))
        interpolated_timeseries_collect.sort_values(['subjID', 'timepoint'])
        # enforce column order
        interpolated_timeseries_collect = interpolated_timeseries_collect[['subjID', 'timepoint', 'dim1', 'dim2']]

        # cluster the timeseries here, but we aren't doing that rn

        # visualize the trajectories thru word embedding space
        # this function takes a stacked set of trajectories, one set of trajectories per subject
        ts_plot_format = np.asarray([interpolated_timeseries_collect[interpolated_timeseries_collect.subjID==s].iloc[:, -2:].values for s in interpolated_timeseries_collect.subjID.unique()])
        # should be in the shape of subjects x timepoints x embedding dimensions
        
        labels = [0 for i in range(ts_plot_format.shape[0])]

        # calculate what the xlim and ylim should be for each video
        xlim = (ts_plot_format[:, :, 0].min() - 3, ts_plot_format[:, :, 0].max() + 3)
        ylim = (ts_plot_format[:, :, 1].min() - 3, ts_plot_format[:, :, 1].max() + 3)

        # get first 10 frequent words
        top_words = word_frequencies.iloc[:10, 0].values
        top_words_w_embedding = embeddings_reduced_df[embeddings_reduced_df.preprocessed.isin(top_words)]

        plot_timeseries(ts_plot_format, labels, top_words = top_words_w_embedding, OUTPUT_PATH=os.path.join(OUTPUT_PATH, it), xlim = xlim, ylim = ylim)


















        