import argparse
import datetime
import multiprocessing as mp
import os
from functools import partial
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
import tweepy
from tweepy import TweepyException

import sys
sys.path.append("..")
from config import *
from utils.utils import only_directories

DEBUG = False


def extract_user_features(raw_user_json):
    if type(raw_user_json) == dict:
        user_features = {key: raw_user_json[key] for key in KEYS}

    else:
        user_features = {key: raw_user_json._json[key] for key in KEYS}

    user_features['created_at'] = datetime.datetime.strptime(user_features['created_at'],
                                                             "%a %b %d %H:%M:%S %z %Y").timestamp()
    return user_features


def crawl_and_extract_user_features(filename, dataset_name, api, stats_all, args):
    dataset_full_path = FAKE_NEWS_DATA + "/" + dataset_name
    old_user_df_fullname = dataset_full_path + "/" + filename + '/old_user.tsv'
    new_user_df_fullname = dataset_full_path + "/" + filename + '/new_user.tsv'

    #if not args.overwrite and os.path.exists(new_user_df_fullname):
    #    print(f"\t{filename} SKIPPED")
    #   return

    if os.path.exists(old_user_df_fullname):
        old_users_df = pd.read_csv(old_user_df_fullname, sep='\t').astype({'id': 'int64'})
    else:
        print(f"\t{filename} does NOT exist")
        return

    user_ids = old_users_df.id.astype('int64').to_list()

    old_users_df.set_index("id", inplace=True)

    all_user_features = []
    n_lost, n_user_feature_from_original_data = 0, 0

    n_iters = int(np.ceil(len(user_ids) / 100))

    for i in range(n_iters):
        batch_user_ids = user_ids[i * TWEEPY_MAX_BATCH:(i + 1) * TWEEPY_MAX_BATCH]

        try:

            # Crawl user feature from twitter API
            # Failures are usually due to "Account suspended"
            # or "User not found"
            raw_user_objects = api.lookup_users(user_id=batch_user_ids)

            all_user_features += [extract_user_features(object) for object in raw_user_objects]

        except TweepyException as e:
            continue
            #print(f"\t{e.api_messages[0]}")

    n_total = len(user_ids)
    users_ids_with_features = [user['id'] for user in all_user_features]
    users_ids_feature_lost = list(set(user_ids) - set(users_ids_with_features))
    n_lost = len(users_ids_feature_lost)

    new_users_df = pd.DataFrame(all_user_features)\

    if not new_users_df.empty:
        new_users_df = new_users_df.astype({'id': 'int64'}).set_index('id')

    if users_ids_feature_lost != []:
        new_users_df = new_users_df.append(old_users_df.loc[users_ids_feature_lost].drop(USER_IDENTITY_KEYS, axis=1))

    n_lost_after_consult = (new_users_df.notna().sum(axis=1) < 5).sum(axis=0)

    n_user_feature_from_original_data = n_lost - n_lost_after_consult

    print(
        f"\t{filename}: Lost: {n_lost_after_consult} ({float(n_lost_after_consult) / n_total * 100 : .2f}%) | Original feat.: {n_user_feature_from_original_data} ({float(n_user_feature_from_original_data) / n_total * 100 : .2f}%) | Total: {n_total}")

    stats = {
        'id': filename,
        "lost": n_lost,
        "original": n_user_feature_from_original_data,
        "total": n_total
    }
    stats_all += [stats]

    new_users_df = new_users_df.join(old_users_df[USER_IDENTITY_KEYS], on='id').reset_index()
    new_users_df.to_csv(new_user_df_fullname, sep='\t', index=False)

    return n_lost, n_user_feature_from_original_data, n_total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--overwrite', action="store_true", help='Whether to overwrite new_users.tsv')

    args = parser.parse_args()

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    politifact_fake = only_directories(POLITIFACT_FAKE)
    politifact_real = only_directories(POLITIFACT_REAL)
    gossipcop_fake = only_directories(GOSSIPCOP_FAKE)
    gossipcop_real = only_directories(GOSSIPCOP_REAL)

    data_list = {
        #POLITIFACT_REAL_NAME: politifact_real,
        #POLITIFACT_FAKE_NAME: politifact_fake,
        GOSSIPCOP_FAKE_NAME: gossipcop_fake,
        GOSSIPCOP_REAL_NAME: gossipcop_real
    }

    manager = Manager()

    print(f"Using {str(mp.cpu_count())} processors")

    # dataset_name is "politifact" or "gossipcop"
    for dataset_name, dataset_list in DATASET_NAMES.items():
        print("#" * 30 + f"\n# Processing {dataset_name}\n" + "#" * 30 + "\n")
        print(dataset_list)
        stats_all = manager.list()

        job_list = []
        for dataset in dataset_list:
            for filename in data_list[dataset]:
                job_list += [[filename, dataset]]
        pool = Pool(processes=mp.cpu_count())

        if True:
            #job_list = [["politifact319", POLITIFACT_REAL_NAME]]
            job_list = [["gossipcop-2116458", GOSSIPCOP_FAKE_NAME]]
            pool = Pool(processes=1)
        partial_process_example = partial(crawl_and_extract_user_features, api=api, stats_all=stats_all, args=args)

        pool.starmap(partial_process_example, job_list)

        user_feature_stats_df = pd.DataFrame(stats_all._getvalue())
        if not user_feature_stats_df.empty:
            user_feature_stats_df.to_csv(FAKE_NEWS_DATA + f"/{dataset_name}_user_feature_stats.tsv", sep='\t', index=False)

            n_user_feature_lost = user_feature_stats_df.get('lost').sum()
            n_user_feature_from_original_data = user_feature_stats_df.get('original').sum()
            n_total = user_feature_stats_df.get('total').sum()

            print("#" * 30 + f"\n# Summary {dataset_name}")
            print(
                f"# Lost: {n_user_feature_lost} ({float(n_user_feature_lost) / n_total * 100 : .2f}%) | Original feat.: {n_user_feature_from_original_data} ({float(n_user_feature_from_original_data) / n_total * 100 : .2f}%) | Total: {n_total}")
        else:
            print("#" * 30 + f"\n# Summary {dataset_name}")
            print(f"# SKIP ALL {dataset_name}\n" + "#" * 30 )

        print("#" * 30 + "\n")
        if DEBUG:
            break


if __name__ == "__main__":
    main()
