#!/usr/bin/env python
# coding: utf-8

# In[16]:

# Use conda environment "hugging"

import json
from spellchecker import SpellChecker
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity

import os
import sys
import pandas as pd
from tqdm import tqdm


def parse_data(dataset_id):
    '''
    dataset_id: a string corresponding to the study name encoded in the task and thus the database
    '''
    json_file = '../../download_data/data/%s' % dataset_id

    with open(json_file, 'r') as f:
        data = json.load(f)

    out_dir = 'processed_data/%s' % dataset_id
    if os.path.exists(out_dir):
        print('Output directory already exists, exiting.')
        sys.exit(1)
    os.makedirs(out_dir)

    timeseries_data = []
    ratings = []
    demos = []
    final_impress = []
    feedback = []
    participants = data['__collections__']['studies'][dataset_id]['__collections__']['participants']

    for p in tqdm(participants.keys()):
        # !!!!!!!!
        if (p=='jac_8_8'):
            continue
        if (p=='test'):
            continue
        # !!!!!!!!

        current_data = participants[p]
        # try parsing demographics
        # if demographics are missing, consider the task unfinished
        try:
            demo_data = current_data['demographics']
        except:
            print('Demos not completed for ID %s, continuing to next participant.' % p)
            continue

        demos.append([p, demo_data['education'][0], demo_data['gender'][0], demo_data['race'][0], demo_data['age'][0], demo_data['sexOrientation'][0]])

        # parse feedback
        feedback_data = current_data['feedback']
        feedback.append([p, feedback_data])

        # parse nlp responses
        # we will extract all video keys (AKA titles of videos watched)
        nlp_data = current_data['__collections__']['video_responses']
        for vid in nlp_data.keys():
            current_ratings = nlp_data[vid]['ratings']
            # add ratings data
            ratings.append([p, vid, current_ratings['competence'], current_ratings['warmth'], current_ratings['conscientiousness'], current_ratings['openness'], current_ratings['neuroticism'], current_ratings['extroversion'], current_ratings['agreeableness']])
            # parse adjectives
            for all_d in nlp_data[vid]['all_descriptors']:
                timeseries_data.append([p, vid, all_d['name'], all_d['timestamp']])
            # parse final impression
            for final_d in nlp_data[vid]['final_descriptors']:
                final_impress.append([p, vid, final_d['name'], final_d['timestamp']])        

        

    
        demos_df = pd.DataFrame(demos, columns = ['subjID', 'education', 'gender', 'race', 'age', 'sex_orientation'])
        demos_df.to_csv(os.path.join(out_dir, 'demos.csv'))

        ratings_df = pd.DataFrame(ratings, columns = ['subjID', 'video', 'competence', 'warmth', 'conscientiousness', 'openness', 'neuroticism', 'extroversion', 'agreeableness'])
        ratings_df.to_csv(os.path.join(out_dir, 'ratings.csv'))

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # term_index is the column of the term and count_index is the
        # column of the term frequency
        sym_spell.load_dictionary("utilities/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

        timeseries_df = pd.DataFrame(timeseries_data, columns = ['subjID', 'video', 'adjective', 'timepoint'])
        # expand the entries that contain commas
        timeseries_df['adjective'] = timeseries_df.adjective.str.split(',')
        timeseries_df = timeseries_df.explode('adjective') 

        # correct the spelling of entries
        #timeseries_df['corrected'] = [spell.correction(word) for word in list(timeseries_df.adjective)]
        #timeseries_df['corrected'] = [TextBlob(word).correct() for word in list(timeseries_df.adjective)]
        timeseries_df['corrected'] = [sym_spell.word_segmentation(word).corrected_string for word in list(timeseries_df.adjective)]


        timeseries_df.to_csv(os.path.join(out_dir, 'timeseries.csv'))

        # ---------

        final_impress_df = pd.DataFrame(final_impress, columns = ['subjID', 'video', 'adjective', 'timepoint'])
        print(final_impress_df[final_impress_df.adjective.isna()])
        
        final_impress_df['adjective'] = final_impress_df.adjective.str.split(',')
        print(final_impress_df[final_impress_df.adjective.isna()])
        final_impress_df = final_impress_df.explode('adjective')
        
        #final_impress_df['corrected'] = [TextBlob(word).correct() for word in list(final_impress_df.adjective)]
        final_impress_df['corrected'] = [sym_spell.word_segmentation(word).corrected_string for word in list(final_impress_df.adjective)]

        final_impress_df.to_csv(os.path.join(out_dir, 'final_impressions.csv'))

        feedback_df = pd.DataFrame(feedback, columns = ['subjID', 'comment'])
        feedback_df.to_csv(os.path.join(out_dir, 'feedback.csv'))

if __name__ == '__main__':
    parse_data(sys.argv[1], sys.argv[2])

