#from scripts.parse_data import parse_data
#from scripts.describe_and_clean import describe_and_clean
#from scripts.get_embeddings import get_embeddings   
#from scripts.model_global_topics import model
from scripts.final_impressions_ratings_rsa import run_rsa
import sys


def main(dataset_id):
    #print('Parsing data for %s' % dataset_id)
    #parse_data(dataset_id)

    #print('Cleaning data for %s' % dataset_id)
    #describe_and_clean(dataset_id)

    # print('Getting embeddings for %s' % dataset_id)
    # get_embeddings(dataset_id)

    # print('Modeling data for %s' % dataset_id)
    # model(dataset_id)

    run_rsa(dataset_id)

if __name__ == '__main__':
    if sys.argv[1]:
        print('Analyzing dataset %s' % sys.argv[1])
        main(sys.argv[1])
    else:
        print('Please specify a dataset ID')
        sys.exit(1)