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
from hdbscan import HDBSCAN

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from bokeh.plotting import figure, output_file, save

import os

# COMPONENTS
# 0. Preprocess the data using standard NLP techniques
# 1. Model topics across all videos
# 2. Analyze trajectories inside the global space

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
# topic modeling
##############################

def model_topics(document_list):
    # embedding model
    ft = api.load('fasttext-wiki-news-subwords-300')

    # dimensionality reduction
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # clustering
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)  

    # count vectorizer
    vectorizer_model = CountVectorizer(stop_words='english')

    # c-tf-idf
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(
        embedding_model=ft,          # Step 1 - Extract embeddings
        umap_model=umap_model,                    # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model)
    
    topics, probs = topic_model.fit_transform(document_list)

    # COULD ADD: visualize the raw word embedding space

    return topics, probs, topic_model
    


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
    print(max(ts[:, 0]))

    # if two timepoints are the same, add a small amount of noise to the second one
    # this is to prevent the interpolation function from failing
    #for i in range(ts.shape[0]-1):
    #    if ts[i, 0] == ts[i+1, 0]:
    #        ts[i+1, 0] += 1

    # note: that some entries in the timeseries are by chance so close to the end of the video
    # that they are being rounded to 500, meaning that there is nothing to interpolate between that
    # and the final impression. If there is such an impression, manually set its timepoint to 499
    if ts[-2, 0] == num_points:
        ts[-2, 0] = num_points - 1
    
    out_array = []
    for i in range(ts.shape[0]-1):
        num_points_interpolate = int(ts[i+1, 0] - ts[i, 0])
        print(ts[i+1, 0])
        print(ts[i, 0])
        print(num_points_interpolate)
        interpolated = interpolate(ts[i,1:], ts[i+1,1:], num_points_interpolate)
        # add global time timepoints
        timepoints = np.arange(ts[i, 0], ts[i+1, 0], 1)
        interpolated = np.insert(interpolated, 0, timepoints, axis=1)
        out_array.extend(interpolated)

    out_array = np.asarray(out_array)
    print(out_array.shape)
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
            plt.text(row['dim1'], row['dim2'], row['topic_label'], fontsize=8, color='black', alpha=0.5)   


    # create colors equal to the number of unique labels
    colors = sb.color_palette("hls", len(np.unique(labels)))

    lines = [ax.plot([], [], color = colors[labels[i]], alpha=0.2)[0] for i in range(ts.shape[0])]
    points = [ax.plot([], [], 'o', color = colors[labels[i]], alpha=0.6)[0] for i in range(ts.shape[0])]

    def animate(num, data, lines):
        for it, line in enumerate(lines):
            if num < 50:
                line_data = data[it]
                line.set_data(line_data[:num, 0:2].T)

                point = points[it]
                point.set_data(line_data[num, 0:2].T)
            else:
                line_data = data[it]
                line.set_data(line_data[num-50:num, 0:2].T)

                point = points[it]
                point.set_data(line_data[num, 0:2].T)

    ani = matplotlib.animation.FuncAnimation(
        fig, animate, NUM_ITER, fargs=(ts, lines), interval=10)

    writervideo = matplotlib.animation.FFMpegWriter(fps=30)
    ani.save(os.path.join(OUTPUT_PATH, 'timeseries.mp4'), writer=writervideo)
    plt.clf()

def plot_umap_interactive(umapper, labels, OUTPUT_PATH=None):
    hover_data = pd.DataFrame()
    hover_data['label'] = labels

    p = umap.plot.interactive(umapper, hover_data = hover_data, tools=['pan', 'wheel_zoom', 'box_zoom', 'reset'])

    output_file(filename=os.path.join(OUTPUT_PATH, "all_global_topics.html"), title="Static HTML file")
    save(p)


##############################
# main function
##############################

def model(dataset_id):
    # load data that contains embeddings

    DERIVATIVES_PATH, OUTPUT_PATH = create_output_folders(dataset_id)

    timeseries_cleaned = pd.read_csv('embeddings/%s/timeseries_w_embeddings.csv' % dataset_id, index_col=None)
    final_impressions_cleaned = pd.read_csv('embeddings/%s/final_impressions_w_embeddings.csv' % dataset_id, index_col = None)
    all_data = pd.concat([timeseries_cleaned.reset_index(drop=True), final_impressions_cleaned.reset_index(drop=True)], axis=0)

    # create global topic space
    topics, probs, topic_model = model_topics(list(all_data['preprocessed']))

    # visualize and save the global topic space
    all_topic_embeddings = topic_model.topic_embeddings_
    all_topic_labels = [topic_model.topic_labels_[i] for i in topic_model.topic_labels_.keys()]

    all_data['topic'] = topics
    # all_data['topic_label'] = np.asarray(all_topic_labels)[topics]

    topic_reducer = umap.UMAP(metric='cosine', n_neighbors=15, min_dist = 0.1)
    topics_reduced = topic_reducer.fit_transform(all_topic_embeddings)

    plot_umap_interactive(topic_reducer, all_topic_labels, OUTPUT_PATH=OUTPUT_PATH)

    # merge the reduced topics onto all_data

    topics_reduced_df = pd.DataFrame(topics_reduced, columns=['dim1', 'dim2'])
    topics_reduced_df['topic_label'] = all_topic_labels
    topics_reduced_df['topic'] = [i for i in range(-1, len(all_topic_labels)-1)]

    all_data_w_topics = all_data[['subjID', 'video', 'adjective', 'timepoint', 'corrected', 'preprocessed', 'topic']].merge(topics_reduced_df, on='topic', how='inner')
    
    all_data_w_topics.to_csv(os.path.join(DERIVATIVES_PATH, 'topics_reduced_no_timepoint_avg.csv'))

    # calculate what the xlim and ylim should be for each video for trajectory viz
    xlim = (topics_reduced_df['dim1'].min() - 3, topics_reduced_df['dim1'].max() + 3)
    ylim = (topics_reduced_df['dim2'].min() - 3, topics_reduced_df['dim2'].max() + 3)

    for it, df in tqdm(all_data_w_topics.groupby('video')):
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

        # identify the most frequent topics
        topic_frequencies = df['topic_label'].value_counts().reset_index(name='counts').sort_values(by='counts', ascending=False)
        topic_frequencies.to_csv(os.path.join(DERIVATIVES_PATH, it, 'topic_frequencies.csv'))

        # identify emergent content, defined by the words entered at the end, minus the words entered during the timeseries
        max_timepoint = df.timepoint.max()
        timeseries_content = df[df.timepoint != max_timepoint].preprocessed
        emergent_content = df[df.timepoint == max_timepoint]
        emergent_content_words = emergent_content[~emergent_content.preprocessed.isin(timeseries_content)]
        emergent_count = emergent_content_words.preprocessed.value_counts().reset_index(name='counts')
        emergent_count.to_csv(os.path.join(DERIVATIVES_PATH, it, 'emergent_content.csv'))

        # identify the moments eliciting most responding
        sb.histplot(data=df[(df.timepoint != 0) & (df.timepoint != max_timepoint)], x="timepoint", kde=True, bins = 50)
        plt.savefig(os.path.join(DERIVATIVES_PATH, it, 'timepoint_histogram.png'))
        plt.clf()
                        
        # average together the embeddings for identical timepoints
        df_timepoint_avg = df.groupby(['subjID', 'timepoint']).agg({'dim1': 'mean', 'dim2': 'mean', 'preprocessed': lambda x: ', '.join(x)}).reset_index()
        df_timepoint_avg.to_csv(os.path.join(DERIVATIVES_PATH, it, 'topics_reduced_w_timepoint_avg.csv'))

        # interpolate between timepoints
        interpolated_timeseries_collect = pd.DataFrame()
        NUM_POINTS = 500
        for it_s, s in df_timepoint_avg.groupby('subjID'):
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

        # identify top topics
        top_topics = topic_frequencies.iloc[:10, 0]
        top_topics_w_embedding = topics_reduced_df[topics_reduced_df.topic_label.isin(top_topics)]
        top_topics_w_embedding = top_topics_w_embedding[['topic_label', 'dim1', 'dim2']]

        # plot_timeseries(ts_plot_format, labels, top_words = top_words_w_embedding, OUTPUT_PATH=os.path.join(OUTPUT_PATH, it), xlim = xlim, ylim = ylim)
        plot_timeseries(ts_plot_format, labels, top_words=top_topics_w_embedding, OUTPUT_PATH=os.path.join(OUTPUT_PATH, it), xlim = xlim, ylim = ylim)


















        