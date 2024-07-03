"""
Count the number of posts, users, and keywords
"""
import argparse
import configparser
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import os.path as osp
import pandas as pd

from utils import utils
from utils.utils import get_root_dir, MAPPING, MAPPING_LABELS, print_msg

"""Specify the root directory to the raw dataset here"""
raw_data_dir = "E:\\fake_news_data"


def run(filename, all_tweets_d, all_replies_d, labels_d, stats_d, args, config):
    # filename = "politifact14258"
    print(filename)

    if filename in all_tweets_d:
        n_tweets = len(all_tweets_d[filename])
    else:
        n_tweets = 0

    if filename in all_replies_d:
        n_replies = len(all_replies_d[filename])
    else:
        n_replies = 0

    label = labels_d[filename]

    example_name = osp.join(raw_data_dir, f"{args.dataset}_{'fake' if label == 1 else 'real'}", filename)
    path_users_df = osp.join(example_name, "new_user.tsv")
    if not osp.exists(path_users_df):
        n_users = 0
    else:
        users_df = pd.read_csv(path_users_df, sep='\t', dtype={
            'id': np.int64
        }, float_precision='high')
        n_users = len(users_df)

        for i, user_meta in users_df.iterrows():
            interactions_user[label][int(user_meta.id)] += [filename]

    path_tweet_df = osp.join(example_name, "tweets_retweets_comments.tsv")
    if osp.exists(path_tweet_df):

        try:
            tweet_retweet_comment_df = pd.read_csv(path_tweet_df, sep='\t', dtype={
                'id': np.int64
            }, encoding="utf-8")
        except:
            tweet_retweet_comment_df = pd.read_csv(path_tweet_df, sep='\t', dtype={
                'id': np.int64
            }, encoding='unicode_escape')
        # user_ids = tweet_retweet_comment_df.user_id.to_list()
        for i, post in tweet_retweet_comment_df.iterrows():
            type = MAPPING[post.type]
            stats_posts_user[label][type][post.user_id] += [filename]

        """
        Count the total #interactions. Tweet and reply are counted separately
        """
        user_ids = np.array(tweet_retweet_comment_df.user_id.to_list(), dtype=np.int64)
        c = Counter(list(user_ids))
        stats_posts_interactions[label].update(c)

        """
        Count the total number of interactions. If a user tweets and replies in a news, they are counted as once
        """
        user_ids = np.array(tweet_retweet_comment_df.user_id.to_list(), dtype=np.int64)
        c = Counter(set(list(user_ids)))
        stats_posts_interactions_unique[label].update(c)

    stats = {
        "Tweet": n_tweets,
        "Reply": n_replies,
        "Users": n_users
    }

    stats_d[filename] = stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--config_file', type=str, required=True)

    # ------------------------------------------
    # Arguments for generating user embeddings
    # ------------------------------------------

    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--only_tweet_stats', action='store_true',
                        help='Only get the statistics of the number of tweets')
    parser.add_argument('--get_user_embeddings', action='store_true')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--in_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    args.dataset = "politifact" if args.config_file == "P.ini" else "gossipcop"

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processed_dir = get_root_dir()
    stats_d = {}

    interactions_user = {
        1: defaultdict(list),
        0: defaultdict(list)
    }

    stats_posts_user = {
        # fake
        1: {
            "tweet"  : defaultdict(list),
            "retweet": defaultdict(list),
            "reply"  : defaultdict(list)
        },  # real
        0: {
            "tweet"  : defaultdict(list),
            "retweet": defaultdict(list),
            "reply"  : defaultdict(list)
        }
    }

    stats_posts_interactions = {
        1: Counter(),
        0: Counter()
    }
    stats_posts_interactions_unique = {
        1: Counter(),
        0: Counter()
    }

    all_tweets_d, all_replies_d, _ = torch.load(osp.join(processed_dir, f"{args.dataset}_tweets.pt"))
    config = configparser.ConfigParser()
    path_config_file = osp.join("..", "FineReasoning", "kgat", "config", args.config_file)
    if osp.exists(path_config_file):
        print(f"Loading {args.config_file}")
        config.read(path_config_file)
    else:
        raise Exception("Cannot find config file")
    args.dataset = config["KGAT"].get("dataset")
    args.user_embed_dim = config["KGAT"].getint("user_embed_dim")

    global_news_article_evidence_d = utils.read_news_article_evidence(args.dataset)
    labels_d = utils.load_labels(args.dataset)


    def stats_df_from_counter(count, columns=None, orient="index"):
        """
        @param count: Counter
        @return: pd.DataFrame
        """
        frequency_df = pd.DataFrame.from_dict(count, orient=orient, columns=columns).fillna(0)
        frequency_df.index = frequency_df.index.astype(np.int64)
        frequency_df = frequency_df.sort_values(frequency_df.columns[0], ascending=False)
        frequency_df.index.rename('user_id', inplace=True)
        return frequency_df


    for i, filename in enumerate(global_news_article_evidence_d):
        run(filename, all_tweets_d, all_replies_d, labels_d, stats_d, args, config)
        # if i >= 40:
        #     break

    stats_df = pd.DataFrame.from_dict(stats_d).transpose()
    print_msg("Count #interactions of each user")

    count_posts_interactions = dict(Counter(stats_posts_interactions[0] + stats_posts_interactions[1]).most_common())

    count_posts_interactions_unique = dict(Counter(stats_posts_interactions_unique[0] + stats_posts_interactions_unique[1]).most_common())

    count_posts_user = {
        # fake
        1: {
            "tweet"  : defaultdict(dict),
            "retweet": defaultdict(dict),
            "reply"  : defaultdict(dict)
        },

        # real
        0: {
            "tweet"  : defaultdict(dict),
            "retweet": defaultdict(dict),
            "reply"  : defaultdict(dict)
        }
    }

    count_posts_user_unique = {
        # fake
        1: {
            "tweet"  : defaultdict(dict),
            "retweet": defaultdict(dict),
            "reply"  : defaultdict(dict)
        },  # real
        0: {
            "tweet"  : defaultdict(dict),
            "retweet": defaultdict(dict),
            "reply"  : defaultdict(dict)
        }
    }

    """
    count_posts_user: number of post of post_type a user has participated in
    count_posts_user_unique: number of post of post_type a user has participated in
    
    label: 0 or 1
    """

    POST_TYPES = ["tweet", "retweet", "reply"]
    for label in [1, 0]:

        """post_type: tweet, retweet, or reply"""
        for post_type in POST_TYPES:

            """post of a specific type for user_id"""
            for user_id, posts_of_one_user in stats_posts_user[label][post_type].items():
                count_posts_user[label][post_type][user_id] = len(posts_of_one_user)

                """Unique posts a user has participated in"""
                posts_unique = set(posts_of_one_user)
                count_posts_user_unique[label][post_type][user_id] = len(posts_unique)

    pd.set_option("float_format", '{:.2f}'.format)
    pd.set_option("display.max_columns", 10)
    percentiles = np.concatenate([np.arange(0.1, 0.9, 0.1), np.arange(0.9, 1.0, 0.01)])
    d = {
        1: None,
        0: None
    }

    count_user_fake = {k: len(v) for k, v in interactions_user[1].items()}
    count_user_real = {k: len(v) for k, v in interactions_user[0].items()}

    count_user_fake_df = stats_df_from_counter(count_user_fake, columns=["num_fake"])
    count_user_real_df = stats_df_from_counter(count_user_real, columns=["num_real"])
    count_user_df = count_user_fake_df.join(count_user_real_df, on="user_id", how="outer").fillna(0)
    count_user_df['total'] = count_user_df["num_fake"] + count_user_df["num_real"]

    """Count #interacted filenames (ANY type)"""
    assert len(count_posts_interactions) == len(count_posts_interactions_unique)
    non_uniques = stats_df_from_counter(count_posts_interactions, columns=['non_unique'])
    uniques = stats_df_from_counter(count_posts_interactions_unique, columns=['unique'])

    for label in [1, 0]:
        df = stats_df_from_counter(dict(count_posts_user[label]), orient="columns").fillna(0)

        """df_unique: #news a user participated"""
        df_unique_one_label = stats_df_from_counter(dict(stats_posts_interactions_unique[label]), columns=[f'num_{MAPPING_LABELS[label]}'])
        df_unique_both_label = stats_df_from_counter(dict(stats_posts_interactions_unique[0] + stats_posts_interactions_unique[1]), columns=[f'num_total'])
        df_all = df_unique_both_label.join(df_unique_one_label, how="outer").fillna(0)
        df_all = df_all.join(df, how="outer").fillna(0)
        df_all = df_all.astype(np.int64)

        print_msg(f"{args.dataset} - {MAPPING_LABELS[label]}")
        print(df_all.describe(percentiles))

    df = uniques.merge(non_uniques, on="user_id", how="outer")
    print(df.describe(percentiles))

    print(stats_df.describe(percentiles))

    print_msg(args.dataset)
    print(count_user_df.describe(percentiles))
    # print(count_user_real_df.describe(percentiles))

    stats_df.to_csv(osp.join("outputs", f"{args.dataset}_stats_d.tsv"), sep="\t")
    print("Done!")
