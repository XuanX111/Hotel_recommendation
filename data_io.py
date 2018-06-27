import csv
from operator import itemgetter
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def get_paths():
    response_folder = '~/Documents/Metis/project5_2/'
    response_folder = os.path.expanduser(response_folder)

    return response_folder

def read_desin():
    destin_path = get_paths()+'destinations.csv'
    return pd.read_csv(destin_path)

def read_train():
    train_path = get_paths()+'train.csv'
    return pd.read_csv(train_path)

def read_test():
    test_path = get_paths() + 'test.csv'
    return pd.read_csv(test_path)

def save_dataframe(df):
    out_path = get_paths()+ 'dataframe.pkl'
    df.to_pickle(out_path, compression='infer', protocol=4)

def save_model(model,file):
    out_path = get_paths() + f'{file}.pkl'
    print(out_path)
    pickle.dump(model,open(out_path, 'wb'))

def load_model(file):
    in_path = get_paths()+f'{file}.pkl'
    return pickle.load(open(in_path,'rb'))

def write_submission(recommendations, submission_file=None):
    if submission_file is None:
        submission_path = get_paths()["submission_path"]
    else:
        path, file_name = os.path.split(get_paths()["submission_path"])
        submission_path = os.path.join(path, submission_file)
    rows = [(srch_id, prop_id)
        for srch_id, prop_id, rank_float
        in sorted(recommendations, key=itemgetter(0,2))]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    writer.writerow(("SearchId", "PropertyId"))
    writer.writerows(rows)

