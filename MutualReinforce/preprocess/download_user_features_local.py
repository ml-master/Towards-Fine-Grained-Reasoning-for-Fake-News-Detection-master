import argparse
import datetime
import os
import os.path as osp
import pickle
import numpy as np
import pandas as pd
#import tweepy
#from tweepy import TweepyException
from multiprocessing import Manager

from config import *

def extract_user_features(raw_user_json):
    #从文档中提取？
    if type(raw_user_json) == dict:
        user_features = {key: raw_user_json[key] for key in KEYS}

    else:
        user_features = {key: raw_user_json._json[key] for key in KEYS}

    user_features['created_at'] = datetime.datetime.strptime(user_features['created_at'], "%a %b %d %H:%M:%S %z %Y").timestamp()
    return user_features

def crawl_and_extract_user_features(api, user_ids, output_dir, example_name, global_user_features_d):
    all_user_features = []
    n_user_feature_lost, n_user_feature_from_original_data = 0, 0

    for user_id in user_ids:

        # We use the example name as the dummy root node (e.g. "politifact11773")
        # So just ignore them

        try:

            # Crawl user feature from twitter API
            # Failures are usually due to "Account suspended"
            # or "User not found"


            raw_user_json = api.get_user(user_id=user_id)   #如何获得raw_user_json?手动提取？
            user_features = extract_user_features(raw_user_json)
            all_user_features += [user_features]
        except:
            #print(f"\tUser {str(user_id):20} | {e.api_messages[0]}")
            user_features_d = global_user_features_d[example_name]
            if user_id in user_features_d:

                # Cannot get user feature from twitter API
                # Use the original feature in the dataset
                # This should be rare
                all_user_features += [user_features_d[user_id]]
                n_user_feature_from_original_data += 1
            else:
                n_user_feature_lost += 1
    n_total = len(all_user_features) + n_user_feature_lost
    print(f"\t{example_name}: Lost: {n_user_feature_lost} ({float(n_user_feature_lost) / n_total * 100 : .2f}%) | Original feat.: {n_user_feature_from_original_data} ({float(n_user_feature_from_original_data) / n_total * 100 : .2f}%) | Total: {n_total}")

    # new_user.tsv stores all users' features for one example
    # how to get the new_user.tsv?
    df = pd.DataFrame(all_user_features)
    df.to_csv(osp.join(output_dir, example_name, 'new_user.tsv'), sep = '\t', index=False)

    return n_user_feature_lost, n_user_feature_from_original_data, n_total


def crawl_and_extract_user_features_batch(user_ids, output_dir, example_name, global_user_features_d, global_user_features_new_d, old_user_df):
    all_user_features = []
    n_user_feature_lost, n_user_feature_from_original_data = 0, 0

    n_iters = int(np.ceil(len(user_ids) / 100))

    """
    for i in range(n_iters):
        #batch_user_ids = user_ids[i*TWEEPY_MAX_BATCH:(i+1)*TWEEPY_MAX_BATCH]

        # We use the example name as the dummy root node (e.g. "politifact11773")
        # So just ignore them

        try:

            # Crawl user feature from twitter API
            # Failures are usually due to "Account suspended"
            # or "User not found"
            raw_user_objects = api.lookup_users(user_id=batch_user_ids)

            all_user_features += [extract_user_features(object) for object in raw_user_objects]
        except:
            print("Cant get")
        #except TweepyException as e:
        #    print(f"\t{e.api_messages[0]}")
    """
    n_total = len(user_ids)
    users_ids_with_features = [user['id'] for user in all_user_features]
    users_ids_feature_lost = set(user_ids) - set(users_ids_with_features)
    n_user_feature_lost = len(users_ids_feature_lost)

    user_features_d = global_user_features_d[example_name]

    old_user_with_features_df = old_user_df[old_user_df.isna().sum(axis=1) < 10].set_index('id')

    for user_id in users_ids_feature_lost:

        # Cannot get user feature from twitter API
        # Use the original feature in the dataset
        # This should be rare

        try:

            all_user_features += [old_user_with_features_df[user_id].to_dict()]
            print("\t{user_id} uses old features")
            n_user_feature_from_original_data += 1
            n_user_feature_lost -= 1
        except:
            print(f"\t{user_id} feature does NOT exist in retweet.json")

    print(f"\t{example_name}: Lost: {n_user_feature_lost} ({float(n_user_feature_lost) / n_total * 100 : .2f}%) | Original feat.: {n_user_feature_from_original_data} ({float(n_user_feature_from_original_data) / n_total * 100 : .2f}%) | Total: {n_total}")

    global_user_features_new_d[example_name] = all_user_features

    # Empty df
    if not all_user_features == []:

        # new_user.tsv stores all users' features for one example
        df = pd.DataFrame(all_user_features).set_index('id')
        df = df.join(old_user_df[['is_tweeter', 'is_tweeter', 'is_replier']], on='id')
        df.to_csv(osp.join(output_dir, example_name, 'new_user.tsv'), sep = '\t', index=False)

    return n_user_feature_lost, n_user_feature_from_original_data, n_total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--use_batch', action="store_true", help='Whether to use batch download and processing')

    args = parser.parse_args()

    manager = Manager()


    #auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    #auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    #api = tweepy.API(auth, wait_on_rate_limit=True)
    '''
    if args.use_batch:
        client = tweepy.Client(bearer_token=BEARER_TOKEN, consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET, access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET, wait_on_rate_limit=False)
    '''

    # { example_name -> list of user features }
    global_user_features_new_d = manager.dict()

    for path in [POLITIFACT_REAL,  POLITIFACT_FAKE]:
        f = open(path + "\\all_user_ids_and_features.pkl", "rb")    #有了all_user_ids_and_features.pkl
        global_user_ids, global_user_features_d = pickle.load(f)
        f.close()

        global_user_ids = manager.dict(global_user_ids)

        stats_all = manager.list()

        for filename, user_ids in global_user_ids.items():
            print(filename)

            old_user_df_fullname = path + "\\" + filename+'\\old_user.tsv'
            new_user_df_fullname = path + "\\" + filename+'\\new_user.tsv'
            if os.path.exists(old_user_df_fullname):
                old_user_df = pd.read_csv(old_user_df_fullname, sep='\t')

            if os.path.exists(new_user_df_fullname):
                df = pd.read_csv(new_user_df_fullname, sep='\t')
                df = df.to_dict()
                global_user_features_new_d[filename] = df
                continue


            if args.use_batch:
                n_user_feature_lost, n_user_feature_from_original_data, n_total = crawl_and_extract_user_features_batch(user_ids, path, filename, global_user_features_d, global_user_features_new_d, old_user_df)

            else:
                n_user_feature_lost, n_user_feature_from_original_data, n_total = crawl_and_extract_user_features(user_ids, path, filename, global_user_features_d)

            stats = {
                'id': filename,
                "lost": n_user_feature_lost,
                "original": n_user_feature_from_original_data,
                "total": n_total
            }


            stats_all += [stats]


        user_feature_stats_df = pd.DataFrame.from_dict(stats_all)
        user_feature_stats_df.to_csv(path +"\\user_feature_stats.tsv", sep = '\t', index=False)

        n_user_feature_lost = user_feature_stats_df.get('lost').sum()
        n_user_feature_from_original_data = user_feature_stats_df.get('original').sum()
        n_total = user_feature_stats_df.get('total').sum()

        dataset_name = path.split('\\')[-1]
        print("#"*30 + f"\n# Summary {dataset_name}")
        print(f"# Lost: {n_user_feature_lost} ({float(n_user_feature_lost) / n_total * 100 : .2f}%) | Original feat.: {n_user_feature_from_original_data} ({float(n_user_feature_from_original_data) / n_total * 100 : .2f}%) | Total: {n_total}")

        print("#"*30 + "\n")



        f = open(path + "\\new_all_user_features.pkl", "wb")
        pickle.dump(global_user_features_new_d,f)
        f.close()

if __name__ == "__main__":
    main()
