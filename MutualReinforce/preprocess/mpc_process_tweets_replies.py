import argparse
import multiprocessing as mp
import os
import os.path as osp
import random
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import torch

import config

import sys
sys.path.append("..")
from utils import utils

#done

def process_tweet_replies(filename, dataset, all_tweets_d, all_tweets_score_d, all_replies_d, args):
    print(f"\t{filename}")
    dtypes = {
        'root_tweet_id': np.int64,
        'tweet_id': np.int64,
        'root_user_id': np.int64,
        'user_id': np.int64,
    }

    path = osp.join(config.FAKE_NEWS_DATA, dataset, filename, "tweets_retweets_comments.tsv")
    if not osp.exists(path):
        print(f"\t SKIP {filename}: no tweet_retweet_comment.tsv")
        return

    df = pd.read_csv(path, sep='\t', float_precision='high')
    df.fillna({
        'root_user_id': 0,
        'root_tweet_id': 0,
        "text": ""
    }, inplace=True)
    df = df.astype(dtypes, copy=False)
    df.set_index('tweet_id', inplace=True)

    tweet_df = df[df.type == 0]
    retweet_df = df[df.type == 1]
    reply_df = df[df.type == 2]
    retweet_count = retweet_df.groupby('root_tweet_id').count().user_id
    reply_count = reply_df.groupby('root_tweet_id').count().user_id

    indices = retweet_count.index.union(reply_count.index)
    indices = indices.intersection(tweet_df.index)
    tweet_df = tweet_df.loc[indices]
    tweet_score = retweet_count.to_dict()

    all_tweets_d[filename] = tweet_df
    all_replies_d[filename] = reply_df
    all_tweets_score_d[filename] = tweet_score

    args.n_processed += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--dataset', type=str, choices=["politifact", "gossipcop"], default="gossipcop",
                        help='which dataset to use')
    parser.add_argument('--use_mpc', action='store_true', help='Use multiple processing')
    args = parser.parse_args()
    DATASET_NAMES = utils.get_dataset_names(args.dataset)
    data_list = utils.get_data_list()

    utils.print_heading(args.dataset)

    if args.debug:
        job_list = [["politifact15246", config.POLITIFACT_FAKE_NAME],
                    ['politifact11773', config.POLITIFACT_FAKE_NAME]]

    else:
        job_list = []
        # Either X_real or X_fake
        for dataset in DATASET_NAMES[args.dataset]:
            for filename in data_list[dataset]:
                job_list += [[filename, dataset]]

    random.shuffle(job_list)
    if args.use_mpc:
        print("Using MPC")
        manager = Manager()
        pool = Pool(processes=1 if args.debug else mp.cpu_count() - 1)

        all_tweets_score_d = manager.dict()
        all_tweets_d = manager.dict()
        all_replies_d = manager.dict()

        partial_process_example = partial(process_tweet_replies, all_tweets_d=all_tweets_d, all_replies_d=all_replies_d,
                                          all_tweets_score_d=all_tweets_score_d, args=args)
        pool.starmap(partial_process_example, job_list)

        all_tweets_d = all_tweets_d._getvalue()
        all_replies_d = all_replies_d._getvalue()
        all_tweets_score_d = all_tweets_score_d._getvalue()

    else:
        print("Do not use MPC")
        all_tweets_score_d = {}
        all_tweets_d = {}
        all_replies_d = {}
        args.n_processed = 0

        for [filename, dataset] in job_list:
            if filename not in all_tweets_d.keys():
                process_tweet_replies(filename, dataset, all_tweets_d=all_tweets_d, all_replies_d=all_replies_d,
                                  all_tweets_score_d=all_tweets_score_d, args=args)

                if args.n_processed > 0 and args.n_processed % 200 == 0:
                    print(f"Processed {args.n_processed} files. Temporarily saving {len(all_tweets_d)} examples ...")
                    utils.save_tweets(args.dataset, all_tweets_d, all_replies_d, all_tweets_score_d)

    utils.save_tweets(args.dataset, all_tweets_d, all_replies_d, all_tweets_score_d)

    # To load, use:
    # torch.load(os.path.join(config.FAKE_NEWS_DATA, f"{dataset_name}_users.pt"))


if __name__ == "__main__":
    main()
