# get embeddings

import gensim.downloader as api

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import os
import pandas as pd

import numpy as np

# load gensim corpus
ft = api.load('fasttext-wiki-news-subwords-300')
video_lengths = pd.read_csv('utilities/video_lengths.csv')

# get gensim embeddings for each word
def get_gensim_embeddings(words):
    embeddings = []
    for w in words:
        try:
            embeddings.append(ft[w])
        except:
            embeddings.append(np.zeros(300))
    return embeddings

# preprocess the data using standard NLP techniques
def preprocess_words(df):
    # to lowercase
    df['corrected'] = df.corrected.str.lower()
    # remove white space
    df['corrected'] = df.corrected.str.strip()
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    df = df[~df.corrected.isin(stop_words)]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    df['preprocessed'] = df.corrected.apply(lambda x: lemmatizer.lemmatize(x, pos='a'))  

    return df

def get_embeddings(dataset_id):

    # check if embeddings folder exists, if so exit
    if os.path.exists('embeddings/%s' % dataset_id):
        print('Embeddings folder already exists, exiting.')
        return
    os.makedirs('embeddings/%s' % dataset_id)   

     # load data
    timeseries_cleaned = pd.read_csv('cleaned/%s/timeseries_cleaned.csv' % dataset_id)

    final_impressions_cleaned = pd.read_csv('cleaned/%s/final_impressions_cleaned.csv' % dataset_id)  

    # replace the timepoint for final impressions with the video length
    video_lengths_dict = dict(zip(video_lengths['video'], video_lengths['length_sec']))
    final_impressions_cleaned['timepoint'] = final_impressions_cleaned.video.map(video_lengths_dict)

    # for timeseries, if the timepoint is greater than the video length, drop the timepoint
    timeseries_cleaned = timeseries_cleaned[timeseries_cleaned.timepoint <= timeseries_cleaned.video.map(video_lengths_dict)]

    # preprocess words
    timeseries_preproc = preprocess_words(timeseries_cleaned)
    final_impressions_preproc = preprocess_words(final_impressions_cleaned)
    print(timeseries_preproc.shape)

    # get embeddings
    ts_embeddings = get_gensim_embeddings(list(timeseries_preproc['preprocessed']))
    print(len(ts_embeddings))

    final_embeddings = get_gensim_embeddings(list(final_impressions_preproc['preprocessed']))

    # append the embeddings back to the dataframes, then save
    ts_w_embeddings = pd.concat([timeseries_preproc.reset_index(), pd.DataFrame(ts_embeddings).reset_index(drop=True)], axis = 1)
    final_w_embeddings = pd.concat([final_impressions_preproc.reset_index(), pd.DataFrame(final_embeddings).reset_index(drop=True)], axis = 1)

    ts_shape_before = ts_w_embeddings.shape
    final_shape_before = final_w_embeddings.shape

    # remove entries in both that have no embedding
    ts_w_embeddings = ts_w_embeddings[~ts_w_embeddings.iloc[:, -300:].sum(axis=1).eq(0)]
    final_w_embeddings = final_w_embeddings[~final_w_embeddings.iloc[:, -300:].sum(axis=1).eq(0)]

    ts_shape_diff = ts_shape_before[0] - ts_w_embeddings.shape[0]
    final_shape_diff = final_shape_before[0] - final_w_embeddings.shape[0]

    with open('embeddings/%s/shape_diff.txt' % dataset_id, 'w') as f:
        f.write('Timeseries shape difference: %s\n' % ts_shape_diff)
        f.write('Final impressions shape difference: %s\n' % final_shape_diff)

    # save
    ts_w_embeddings.to_csv('embeddings/%s/timeseries_w_embeddings.csv' % dataset_id, index=False)
    final_w_embeddings.to_csv('embeddings/%s/final_impressions_w_embeddings.csv' % dataset_id, index=False)
