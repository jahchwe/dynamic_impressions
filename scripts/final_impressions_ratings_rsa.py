from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr, pearsonr, rankdata
import scipy.cluster.hierarchy as sch

import statsmodels.api as sm

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sb

import numpy as np

from tslearn.metrics import dtw, ctw

from tqdm import tqdm

import itertools

import pickle

def plot_sorted_dm(orig_data, distances, pdist_function, title, outpath):

    if not os.path.exists(os.path.join(outpath, 'dm_viz')):
        os.makedirs(os.path.join(outpath, 'dm_viz'))
    
    outpath = os.path.join(outpath, 'dm_viz')

    orig_data = orig_data.to_numpy()

    X = squareform(distances)
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    print(ind)
    print(list((np.argsort(ind))))
    pd_sorted = [orig_data[i, :] for i in list((np.argsort(ind)))]

    ordered_pdist = squareform(pdist_function(pd_sorted))

    plt.clf()
    plt.imshow(ordered_pdist, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(os.path.join(outpath, '%s.jpg' % title))   
    plt.clf()

def plot_sorted_dm_ts(orig_data, distances, orig_subj_list, pdist_function, title, outpath):
    # this is a little different bc the data are not just a matrix, but a dictionary of timeseries
    # we need to resort the subjID list, then run the pdist function using itertools.combinations

    if not os.path.exists(os.path.join(outpath, 'dm_viz')):
        os.makedirs(os.path.join(outpath, 'dm_viz'))
    
    outpath = os.path.join(outpath, 'dm_viz')

    X = squareform(distances)
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    
    # resort the subjID list based on the clustering
    sorted_subj_list = [orig_subj_list[i] for i in list((np.argsort(ind)))]

    # recalculate the pdist using the sorted subjID list
    ordered_pdist = []
    for (i, j) in tqdm(itertools.combinations(sorted_subj_list, 2)):
        similarity = pdist_function(orig_data[i], orig_data[j])
        ordered_pdist.append(similarity)
    
    ordered_pdist = squareform(ordered_pdist)

    plt.clf()
    plt.imshow(ordered_pdist, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(os.path.join(outpath, '%s.jpg' % title))
    plt.clf()

def init_rsa(dataset_id, load_saved = False, final_metric = 'cosine', timeseries_metric = 'ctw', ratings_metric = 'euclidean', plot_DMs = True, OUTPATH = 'rsa_results'):
    
    if load_saved:
        # load pickled pdists
        try: 
            ratings_pdist = pickle.load(os.path.join(OUTPATH, 'pdist_pickles/ratings_pdist.pkl'))
            ratings_pdist_wc = pickle.load(os.path.join(OUTPATH, 'pdist_pickles/ratings_pdist_wc.pkl'))
            ratings_pdist_big5 = pickle.load(os.path.join(OUTPATH, 'pdist_pickles/ratings_pdist_big5.pkl'))

            final_impressions_pdist = pickle.load(os.path.join(OUTPATH, 'pdist_pickles/final_impressions_pdist.pkl'))   
            
            timeseries_pdist = pickle.load(os.path.join(OUTPATH, 'pdist_pickles/timeseries_pdist.pkl'))
            timeseries_pdist_timepoint_averaged = pickle.load(os.path.join(OUTPATH, 'pdist_pickles/timeseries_pdist_timepoint_averaged.pkl'))

            return ratings_pdist, ratings_pdist_wc, ratings_pdist_big5, final_impressions_pdist, timeseries_pdist, timeseries_pdist_timepoint_averaged
        except:
            print('No pickled pdists found, calculating from scratch.')

     # create folder to save pdists as pickle to prevent recalculating
    if not os.path.exists(os.path.join(OUTPATH, 'pdist_pickles')):
        os.makedirs(os.path.join(OUTPATH, 'pdist_pickles'))

    # load all data sources, and calculate pdists for each
    final_impressions_with_embeddings = pd.read_csv('embeddings/%s/final_impressions_w_embeddings.csv' % dataset_id)
    ratings = pd.read_csv('cleaned/%s/ratings_cleaned.csv' % dataset_id)
    
    # dictionary containing the reduced timeseries for each video
    reduced_timeseries_interpolated = {}
    for v in ratings.video.unique():
        reduced_timeseries_interpolated[v] = pd.read_csv('modeling/%s/derivatives/%s/interpolated_timeseries.csv' % (dataset_id, v))
    
    timeseries_w_embeddings = pd.read_csv('embeddings/%s/timeseries_w_embeddings.csv' % dataset_id)

    # loop through all videos, storing the pdists for each in their own dictionary

    # ratings pdists
    print('Calculating ratings pdists')
    ratings_pdist = {}
    for v in tqdm(ratings.video.unique()):
        v_ratings = ratings[ratings.video == v]
        v_ratings = v_ratings[['subjID', 'competence', 'warmth', 'conscientiousness', 'openness', 'neuroticism', 'extroversion', 'agreeableness']]
        v_ratings = v_ratings.sort_values('subjID')
        distances = pdist(v_ratings.iloc[:, 1:], metric = ratings_metric)
        ratings_pdist[v] = distances

        # plot the sorted distance matrix
        if plot_DMs:
            plot_sorted_dm(v_ratings.iloc[:, 1:], distances, lambda x: pdist(x, metric = ratings_metric), 'ratings_%s' % v, OUTPATH)
    
    print('Calculating ratings (warmth/competence)')
    ratings_pdist_wc = {}
    for v in tqdm(ratings.video.unique()):
        v_ratings = ratings[ratings.video == v]
        v_ratings = v_ratings[['subjID', 'competence', 'warmth']]
        v_ratings = v_ratings.sort_values('subjID')
        distances = pdist(v_ratings.iloc[:, 1:], metric = ratings_metric)
        ratings_pdist_wc[v] = distances

        # plot the sorted distance matrix
        if plot_DMs:
            plot_sorted_dm(v_ratings.iloc[:, 1:], distances, lambda x: pdist(x, metric = ratings_metric), 'ratings_wc_%s' % v, OUTPATH)
    
    print('Calculating ratings (big 5)')
    ratings_pdist_big5 = {}
    for v in tqdm(ratings.video.unique()):
        v_ratings = ratings[ratings.video == v]
        v_ratings = v_ratings[['subjID', 'conscientiousness', 'openness', 'neuroticism', 'extroversion', 'agreeableness']]
        v_ratings = v_ratings.sort_values('subjID')
        distances = pdist(v_ratings.iloc[:, 1:], metric = ratings_metric)
        ratings_pdist_big5[v] = distances

        # plot the sorted distance matrix
        if plot_DMs:
            plot_sorted_dm(v_ratings.iloc[:, 1:], distances, lambda x: pdist(x, metric = ratings_metric), 'ratings_big5_%s' % v, OUTPATH)

    
    # final impressions pdists
    print('Calculating final impressions pdists')
    final_impressions_pdist = {}
    for v in tqdm(final_impressions_with_embeddings.video.unique()):
        v_final = final_impressions_with_embeddings[final_impressions_with_embeddings.video == v]
        # average together the embeddings for the final impressions
        v_final_averaged = v_final.groupby('subjID').mean().reset_index()
        v_final_averaged = v_final_averaged[['subjID'] + list(v_final_averaged.columns[-300:])]
        v_final_averaged = v_final_averaged.sort_values('subjID')
        distances = pdist(v_final_averaged.iloc[:, -300:], metric = final_metric)
        final_impressions_pdist[v] = distances

        # plot the sorted distance matrix
        if plot_DMs:
            plot_sorted_dm(v_final_averaged.iloc[:, -300:], distances, lambda x: pdist(x, metric = final_metric), 'final_impressions_%s' % v, OUTPATH)

        
    # timeseries pdists
    unique_subjects = ratings.subjID.sort_values().unique()

    timeseries_pdist = {}
    print('Calculating timeseries pdists')
    for v in tqdm(ratings.video.unique()):
        print('Calculating timeseries pdists for %s' % v)
        reduced_ts = reduced_timeseries_interpolated[v]
        subj_timeseries = {}
        for subj in reduced_ts.subjID.unique():
            current_ts = reduced_ts[reduced_ts.subjID == subj]
            subj_timeseries[subj] = current_ts.sort_values('timepoint', ascending = True).loc[:, ['dim1', 'dim2']].to_numpy()
    
        ts_pdist_store = []
        for (i, j) in tqdm(itertools.combinations(unique_subjects, 2)):
            if timeseries_metric == 'ctw':
                ts_similarity = ctw(subj_timeseries[i], subj_timeseries[j])
            elif timeseries_metric == 'dtw':
                ts_similarity = dtw(subj_timeseries[i], subj_timeseries[j])
            ts_pdist_store.append(ts_similarity)
        
        timeseries_pdist[v] = ts_pdist_store

        # plot the sorted distance matrix
        if plot_DMs:
            plot_sorted_dm_ts(subj_timeseries, ts_pdist_store, unique_subjects, lambda x, y: ctw(x, y), 'timeseries_%s' % v, OUTPATH)
    
    # calculate pdist for timepoint averaged timeseries embeddings
    print('Calculating timepoint averaged timeseries pdists')
    timeseries_pdist_timepoint_averaged = {}
    for v in tqdm(timeseries_w_embeddings.video.unique()):
        v_ts = timeseries_w_embeddings[timeseries_w_embeddings.video == v]
        # average together the embeddings across time for each subject
        v_ts_averaged = v_ts.groupby('subjID').mean().reset_index()
        v_ts_averaged = v_ts_averaged[['subjID'] + list(v_ts_averaged.columns[-300:])]
        v_ts_averaged = v_ts_averaged.sort_values('subjID')
        distances = pdist(v_ts_averaged.iloc[:, -300:], metric = final_metric)
        timeseries_pdist_timepoint_averaged[v] = distances


    assert len(ratings_pdist) == len(final_impressions_pdist) == len(timeseries_pdist) == len(timeseries_pdist_timepoint_averaged)

    # save the pdists as pickles
    pickle.dump(ratings_pdist, open(os.path.join(OUTPATH, 'pdist_pickles/ratings_pdist.pkl'), 'wb'))
    pickle.dump(ratings_pdist_wc, open(os.path.join(OUTPATH, 'pdist_pickles/ratings_pdist_wc.pkl'), 'wb'))
    pickle.dump(ratings_pdist_big5, open(os.path.join(OUTPATH, 'pdist_pickles/ratings_pdist_big5.pkl'), 'wb'))
    pickle.dump(final_impressions_pdist, open(os.path.join(OUTPATH, 'pdist_pickles/final_impressions_pdist.pkl'), 'wb'))
    pickle.dump(timeseries_pdist, open(os.path.join(OUTPATH, 'pdist_pickles/timeseries_pdist.pkl'), 'wb'))
    pickle.dump(timeseries_pdist_timepoint_averaged, open(os.path.join(OUTPATH, 'pdist_pickles/timeseries_pdist_timepoint_averaged.pkl'), 'wb'))

    return ratings_pdist, ratings_pdist_wc, ratings_pdist_big5, final_impressions_pdist, timeseries_pdist, timeseries_pdist_timepoint_averaged

def calc_rsa(pdist_a, pdist_b, label_a, label_b, title, OUTPATH, permutation = True, comparison_metric = 'spearman'):
    # calculate spearman r between the two pdists
    if comparison_metric == 'spearman':
        r, pval = spearmanr(pdist_a, pdist_b)
    elif comparison_metric == 'pearson':
        r, pval = pearsonr(pdist_a, pdist_b)

    # plot the correlation between the two pdists
    sb.regplot(x = label_a, y = label_b, data = pd.DataFrame({label_a: pdist_a, label_b: pdist_b}), line_kws={'color': 'red'}, scatter_kws={'alpha':0.05})
    plt.xlabel(label_a)
    plt.ylabel(label_b)
    plt.annotate('r = %s' % round(r, 3), xy = (0.5, 0.5), xycoords = 'axes fraction')
    plt.annotate('p = %s' % round(pval, 3), xy = (0.5, 0.4), xycoords = 'axes fraction')
    plt.title(title)
    plt.savefig(os.path.join(OUTPATH, '%s.png' % title))
    plt.close()

def calc_MR_rsa(pdist_y, design_matrix, variable_labels, title, OUTPATH):

    if not os.path.exists(os.path.join(OUTPATH, 'MR')):
        os.makedirs(os.path.join(OUTPATH, 'MR'))

    OUTPATH = os.path.join(OUTPATH, 'MR')   

    X = np.column_stack(design_matrix)
    # standardize the design matrix
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    print(X.shape)
    X = sm.add_constant(X)

    model = sm.OLS(pdist_y, X)
    results = model.fit()

    with open('%s/%s.txt' % (OUTPATH, title), 'w') as f:
        f.write('Multiple regression RSA results\n')
        predictors = ['const'] + variable_labels[1:]
        f.write('Predicting %s from %s\n' % (variable_labels[0], ', '.join(predictors)))
        f.write('NOTE: PREDICTORS HAVE BEEN STANDARDIZED (AKA MEAN SUBTRACTED AND DIV BY STD)\n\n')
        f.write(results.summary().as_text())

def run_rsa(dataset_id, final_metric = 'cosine', timeseries_metric = 'ctw', ratings_metric = 'euclidean', comparison_metric = 'spearman'):
    # create output folder for RSA results
    if not os.path.exists('rsa_results'):
        os.makedirs('rsa_results')

    if not os.path.exists('rsa_results/%s' % dataset_id):
        os.makedirs('rsa_results/%s' % dataset_id)

    # create a folder for the specific variant of the RSA
    variant_id = 'ratings-%s_final-%s_ts-%s' % (ratings_metric, final_metric, timeseries_metric)
    if not os.path.exists('rsa_results/%s/%s' % (dataset_id, variant_id)):
        os.makedirs('rsa_results/%s/%s' % (dataset_id, variant_id))
    
    OUTPATH = 'rsa_results/%s/%s' % (dataset_id, variant_id)

    # calculate the distance matrices for all data sources
    ratings_pdist, ratings_pdist_wc, ratings_pdist_big5, final_impressions_pdist, timeseries_pdist, timeseries_timepoint_avg_pdist = init_rsa(dataset_id, final_metric, timeseries_metric, ratings_metric, plot_DMs = False, OUTPATH = OUTPATH)

    # loop through videos
    '''
    for v in ratings_pdist.keys():
        calc_rsa(ratings_pdist[v], final_impressions_pdist[v], 'ratings', 'final impressions', 'ratings_final_%s' % v, OUTPATH, comparison_metric)
        calc_rsa(ratings_pdist[v], timeseries_pdist[v], 'ratings', 'timeseries', 'ratings_timeseries_%s' % v, OUTPATH, comparison_metric)
        calc_rsa(final_impressions_pdist[v], timeseries_pdist[v], 'final impressions', 'timeseries', 'final_timeseries_%s' % v, OUTPATH, comparison_metric)
    '''
    # calculate multiple regression RSA, predicting final impressions from the timeseries pdist and the timeseries timepoint averaged pdist
    for v in ratings_pdist.keys():
        calc_MR_rsa(final_impressions_pdist[v], [timeseries_pdist[v], timeseries_timepoint_avg_pdist[v]], ['final_impressions', 'timeseries', 'timeseries_timepoint_avg'], 'final_timeseries_MR_%s' % v, OUTPATH)   
        calc_MR_rsa(ratings_pdist[v], [timeseries_pdist[v], timeseries_timepoint_avg_pdist[v]], ['ratings', 'timeseries', 'timeseries_timepoint_avg'], 'ratings_timeseries_MR_%s' % v, OUTPATH)
        calc_MR_rsa(ratings_pdist_big5[v], [timeseries_pdist[v], timeseries_timepoint_avg_pdist[v]], ['ratings_big5', 'timeseries', 'timeseries_timepoint_avg'], 'ratings_big5_timeseries_MR_%s' % v, OUTPATH)
        calc_MR_rsa(ratings_pdist_wc[v], [timeseries_pdist[v], timeseries_timepoint_avg_pdist[v]], ['ratings_wc', 'timeseries', 'timeseries_timepoint_avg'], 'ratings_wc_timeseries_MR_%s' % v, OUTPATH)    