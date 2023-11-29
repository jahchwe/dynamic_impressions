import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from ydata_profiling import ProfileReport
from nltk.corpus import stopwords


def describe_and_clean(dataset_id):
    try:
        timeseries_data = pd.read_csv('processed_data/%s/timeseries.csv' % dataset_id)
        final_impressions = pd.read_csv('processed_data/%s/final_impressions.csv' % dataset_id)
        ratings = pd.read_csv('processed_data/%s/ratings.csv' % dataset_id)
        demos = pd.read_csv('processed_data/%s/demos.csv' % dataset_id)

    except:
        print("No processed timeseries and/or final impressions and/or ratings found. Please run parse_data.py first.")
        exit()
    
    # create descriptive folder if folder doesn't exist
    if not os.path.exists('descriptives'):
        os.mkdir('descriptives')

    # create descriptives folder inside processed_data if folder doesn't exist
    if not os.path.exists('descriptives/%s' % dataset_id):
        os.mkdir('descriptives/%s' % dataset_id)

    # create plots folder inside descriptives if folder doesn't exist
    if not os.path.exists('descriptives/%s/plots' % dataset_id):
        os.mkdir('descriptives/%s/plots' % dataset_id)
    
    # create output txt file in descriptives folder for writing descriptives to
    output_file = open('descriptives/%s/descriptives.txt' % dataset_id, 'w')

    # number of participants
    num_subjs = len(timeseries_data.subjID.unique())
    output_file.write(f'Dataset contains {num_subjs} participants\n')

    #####
    # Create cleaned dataset that ensures that every participant:
    # 
    # 1. Saw all videos
    # 2. Has all first impressions
    # 3. Has all final impressions (at least 5 per video)
    # 4. ~~Contains more than 4 timepoints per video~~ (Not doing this rn, I don't think it's actually necessary
    #####

    # expand entries by white space
    timeseries_data['corrected'] = timeseries_data.corrected.str.split(' ', expand=False)
    timeseries_data = timeseries_data.explode('corrected')

    final_impressions['corrected'] = final_impressions.corrected.str.split(' ', expand=False)   
    final_impressions = final_impressions.explode('corrected')

    # remove stop words
    stopwords_list = set(stopwords.words('english'))
    timeseries_data = timeseries_data[timeseries_data.corrected.isin(stopwords_list) == False]
    final_impressions = final_impressions[final_impressions.corrected.isin(stopwords_list) == False] 


    videos_per_subj = timeseries_data.groupby('subjID')['video'].apply(lambda x: len(set(x)))
    
    # how many have all first impressions?
    timeseries_data.timepoint = np.where(timeseries_data.timepoint < 1, 0, timeseries_data.timepoint)
    # every video should have one, so each subject should have 10 timepoints of 0 at least
    first_impressions_per_subj = timeseries_data[timeseries_data.timepoint==0].groupby(['subjID', 'video'])['timepoint'].count()
    # for each subject, could the number of videos with at least one first impression
    first_impression_agg = first_impressions_per_subj.groupby('subjID').count()

    num_subjs_final = len(final_impressions.subjID.unique())
    num_final = final_impressions.groupby(['subjID', 'video'])['timepoint'].count()
    num_final_more_than_5 = num_final[num_final>4].groupby('subjID').count()
    all_final_impressions = num_final_more_than_5[num_final_more_than_5==10].count()

    # participants who saw all videos
    good_subjs_saw_all = videos_per_subj[videos_per_subj==10].index
    output_file.write(f'Participants who saw all videos: {good_subjs_saw_all.shape}\n')

    # participants who formed first impressions for all videos
    good_subjs_first_impressions = first_impression_agg[first_impression_agg==10].index
    output_file.write(f'Participants who formed first impressions for all videos: {good_subjs_first_impressions.shape}\n')

    # participants who have at least 5 final impressions
    good_subjs_all_final_impressions = num_final_more_than_5[num_final_more_than_5==10].index
    output_file.write(f'Participants who have at least 5 final impressions for all videos: {good_subjs_all_final_impressions.shape}\n')


    intersect_good_subjs = good_subjs_saw_all.intersection(good_subjs_first_impressions)
    intersect_good_subjs = intersect_good_subjs.intersection(good_subjs_all_final_impressions)
    output_file.write(f'Participants who saw all videos, formed first impressions for all videos, and have at least 5 final impressions for all videos: {intersect_good_subjs.shape}\n')

    ####
    # Create cleaned files that exclude subjects that didn't finish the task
    ####
    if not os.path.exists('cleaned'):
        os.mkdir('cleaned')

    if not os.path.exists('cleaned/%s' % dataset_id):
        os.mkdir('cleaned/%s' % dataset_id)

    timeseries_data_clean = timeseries_data[timeseries_data.subjID.isin(intersect_good_subjs)]
    timeseries_data_clean.to_csv('cleaned/%s/timeseries_cleaned.csv' % dataset_id)

    final_impressions_clean = final_impressions[final_impressions.subjID.isin(intersect_good_subjs)]
    final_impressions_clean.to_csv('cleaned/%s/final_impressions_cleaned.csv' % dataset_id)

    ratings_clean = ratings[ratings.subjID.isin(intersect_good_subjs)]
    ratings_clean.to_csv('cleaned/%s/ratings_cleaned.csv' % dataset_id)

    demos_clean = demos[demos.subjID.isin(intersect_good_subjs)]
    demos_clean.to_csv('cleaned/%s/demos_cleaned.csv' % dataset_id)

    ############################################################################################################################
    # TIMESERIES INFO
    ############################################################################################################################

    # how many participants have at least 4 entries for every video in the timeseries?
    timepoints_per_video = timeseries_data_clean.groupby(['subjID','video'])['adjective'].count()
    timepoints_per_video_counted = timepoints_per_video[timepoints_per_video>1].groupby('subjID').count()
    output_file.write(f'Participants who have at least 4 entries for every video in the timeseries: {timepoints_per_video_counted[timepoints_per_video_counted == 10].count()}\n')

    # How many words were entered in total?
    output_file.write('Total number of words entered in timeseries: %s\n' % timeseries_data_clean.shape[0])

    # How many words are unique? What percentage are unique?
    output_file.write('Number of unique words in timeseries: %s\n' % len(timeseries_data_clean.adjective.unique()))
    output_file.write('Percentage of unique words in timeseries: %s\n' % (len(timeseries_data_clean.adjective.unique()) / timeseries_data_clean.shape[0]))

    # How many entries did each participant make on average? Note that this is across all videos.
    output_file.write('Average number of entries per participant: %s\n' % timeseries_data_clean.groupby('subjID')['adjective'].count().mean())

    # What is the average number of entries made for each subject, across all videos?
    entries_per_subject = timeseries_data_clean.groupby(['subjID','video'])['adjective'].count().groupby('subjID').agg(['mean', np.std])
    #entries_per_subject['mean'].plot.bar(yerr=entries_per_subject['std'])
    entries_per_subject['mean'].sort_values(ascending = False).plot.bar()

    plt.title('Avg. entries per subject')
    plt.xticks([])
    plt.savefig('descriptives/%s/plots/avg_entries_per_subject.png' % dataset_id)
    plt.clf()

    # What is the average number of entries made for each video, across all participants?
    entries_per_video = timeseries_data_clean.groupby(['subjID','video'])['adjective'].count().groupby('video').agg(['mean', np.std])
    entries_per_video['mean'].sort_values(ascending = False).plot.bar(yerr=entries_per_video['std'])
    plt.title('Avg. entries per video')
    plt.xticks([])
    plt.savefig('descriptives/%s/plots/avg_entries_per_video.png' % dataset_id)
    plt.clf()

    output_file.write('\n\n')

    ############################################################################################################################
    # FINAL IMPRESSIONS INFO
    ############################################################################################################################

    # How many words were entered in total?
    output_file.write('Total number of words entered in final impressions: %s\n' % final_impressions_clean.shape[0])

    # unique words in final impressions
    output_file.write('Number of unique words in final impressions: %s\n' % len(final_impressions_clean.corrected.unique()))
    output_file.write('Percentage of unique words in final impressions: %s\n' % (len(final_impressions_clean.corrected.unique()) / final_impressions_clean.shape[0]))


    # average number of words added
    entries_per_subject = final_impressions_clean.groupby(['subjID','video'])['corrected'].count().groupby('subjID').agg(['mean', np.std])
    #entries_per_subject['mean'].plot.bar(yerr=entries_per_subject['std'])
    entries_per_subject['mean'].sort_values(ascending = False).plot.bar()

    plt.title('Avg. final entries per subject')
    plt.xticks([])
    plt.savefig('descriptives/%s/plots/avg_entries_per_subject_final.png' % dataset_id)
    plt.clf()


    # average number of words added per video
    entries_per_subject = final_impressions_clean.groupby(['subjID','video'])['corrected'].count().groupby('video').agg(['mean', np.std])
    #entries_per_subject['mean'].plot.bar(yerr=entries_per_subject['std'])
    entries_per_subject['mean'].sort_values(ascending = False).plot.bar(yerr=entries_per_video['std'])

    plt.title('Avg. final entries per subject')
    plt.xticks([])
    plt.savefig('descriptives/%s/plots/avg_entries_per_video_final.png' % dataset_id)
    plt.clf()

    ############################################################################################################################
    # ADDED WORDS INFO
    ############################################################################################################################
    
    # just words that appear in the end, and not anywhere in the timeseries
    # note that words that are added at the end but in the timeseries will have their timestamp still
    added_words = set(final_impressions_clean.corrected.str.lower()) - set(timeseries_data_clean.corrected.str.lower())
    output_file.write(f'Number of words that are unique to being added, as a percentage of unique words entered at the end: {len(added_words)/len(final_impressions.corrected.unique())}\n')

    added_df = final_impressions_clean[final_impressions_clean.corrected.str.lower().isin(added_words)]
    output_file.write(f'Percentage of words that are unique to being added, as a percentage of all words entered at the end: {len(added_df.corrected.unique())/added_df.shape[0]}\n')

    # How many words were added on average for each video?
    added_per_video = added_df.groupby(['subjID','video'])['adjective'].count().groupby('video').agg(['mean', np.std])
    added_per_video['mean'].plot.bar(yerr=added_per_video['std'])
    plt.title('Avg. number of unique words added (not found in timeseries)')
    plt.xticks([])
    plt.savefig('descriptives/%s/plots/avg_added_per_video.png' % dataset_id)
    plt.clf()

    # How many words on average were added by each participant?
    added_per_subj = added_df.groupby(['subjID','video'])['adjective'].count().groupby('subjID').agg(['mean', np.std])
    added_per_subj['mean'].plot.bar(yerr=added_per_subj['std'])
    plt.title('Avg. number of unique words added (not found in timeseries)')
    plt.xticks([])
    plt.savefig('descriptives/%s/plots/avg_added_per_subj.png' % dataset_id)
    plt.clf()

  
    sns.histplot(timepoints_per_video)
    plt.title('Number of timeseries entries, across all videos')
    plt.savefig('descriptives/%s/plots/timepoints_per_video.png' % dataset_id)
    plt.clf()

    ############################################################################################################################
    # RATINGS INFO
    ############################################################################################################################

    ratings_no_subj = ratings_clean.drop(['Unnamed: 0', 'subjID'], axis = 1)
    ratings_long = pd.melt(ratings_no_subj, id_vars = 'video', value_name = 'rating', var_name= 'trait')
    ratings_long.rating = ratings_long.rating.astype(float)

    f, ax = plt.subplots(figsize=(15, 20))
    sns.barplot(data=ratings_long, x='video', y ='rating', hue='trait', orient='v')
    plt.xticks([])
    plt.savefig('descriptives/%s/plots/ratings.png' % dataset_id)
    plt.clf()   

    ############################################################################################################################
    # DEMOGRAPHICS INFO
    ############################################################################################################################

    profile = ProfileReport(demos_clean, title="Demos Report")
    profile.to_file("descriptives/%s/demos_report.html" % dataset_id)

    output_file.close()

