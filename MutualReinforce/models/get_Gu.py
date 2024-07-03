import argparse
import multiprocessing as mp
import os.path as osp
import random
import traceback
from functools import partial
from multiprocessing import Pool, Manager

#from core.cours_import Cours
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("..")
import config
from utils import utils

def convert_str_features_to_numeric(df, filename, column_names=config.KEYS_STR):
    for feature_name in column_names:
        if feature_name not in df.columns:
            print(f"\t{filename}: Missing field {feature_name}")
        new_feature = df[feature_name].fillna("").astype(str).str.len()
        # new_feature = df[feature_name].astype(str).str.len().fillna(0).astype('int64')
        df[f'len_{feature_name}'] = new_feature
        df.drop(feature_name, axis=1, inplace=True)


def get_edge_weight(s_src, s_tgt, args):
    if args.edge_weight_mode == "ratio":
        impact_s_t = (s_src + args.epsilon) / (s_tgt + args.epsilon)
        impact_t_s = 1. / impact_s_t
    elif args.edge_weight_mode == "reversed_ratio":
        impact_s_t = (s_tgt + args.epsilon) / (s_src + args.epsilon)
        impact_t_s = 1. / impact_s_t
    elif args.edge_weight_mode == "product":
        impact_s_t = impact_t_s = (s_src + args.epsilon) * (s_tgt + args.epsilon)
    elif args.edge_weight_mode == "binary":
        impact_s_t = impact_t_s = 1.
    else:
        raise NotImplementedError
    return impact_s_t, impact_t_s


def process_example(filename, dataset_name, labels_d, all_user_feat_d, all_Gu_d, all_tweets_d, all_replies_d,
                    all_tweets_score_d, args):
    '''

    :param filename: e.g. 'politifact13038'
    :param dataset_name: e.g. 'politifact_fake
    :param all_user_feat_d: dictionary of pandas DataFrame to store
    :return:
    '''
    # filename = "politifact14954"

    dataset_full_path = osp.join(config.FAKE_NEWS_DATA, dataset_name)
    example_name = osp.join(dataset_full_path, filename)
    path_users_df = osp.join(example_name, "new_user.tsv")  #new_user.tsv   1
    if not osp.exists(path_users_df):
        print(f"\t SKIP {filename}: no new_user.tsv")
        return

    users_df = pd.read_csv(path_users_df, sep='\t', dtype={
        'id': np.int64
    }, float_precision='high')

    users_df.set_index('id', inplace=True)  #将id列设为索引
    users_df = pd.read_csv(path_users_df)

    tweet_df = all_tweets_d[filename]   #提取tweet和reply的df
    reply_df = all_replies_d[filename]
    tweet_score = all_tweets_score_d[filename]

    tweet_df = pd.concat([tweet_df, reply_df])
    user_ids_keep = list(set(tweet_df.root_user_id) & set(tweet_df.user_id))
    indices = pd.Int64Index(user_ids_keep).intersection(users_df.index)     #取两对象交集

    users_df = users_df.loc[indices]    #提取索引为indices的那一行
    print(f"{filename} | {len(users_df)}")

    if len(users_df.columns) <= 4:
        print(f"\t SKIP {filename}: too few columns")
        return

    # If the user features are empty, remove this example
    if users_df.empty:
        print(f"\t SKIP {filename}: empty new_user.tsv")
        return
    # ------------------------------------------
    # Fill null values
    # ------------------------------------------

    try:
        users_df.fillna({
            "geo_enabled": False,
            "verified": False,
            "followers_count": 0,
            "friends_count": 0,
            "listed_count": 0,
            "statuses_count": 0,
            "favourites_count": 0,
        }, inplace=True)

        if 'created_at' not in users_df.columns:
            users_df[['created_at']] = 1272074231
        users_df.fillna({
            'created_at': 1272074231
        }, inplace=True)

        usr_identity_li = ['is_tweeter', 'is_retweeter', 'is_replier']


        for colname in usr_identity_li:
            users_df.loc[users_df[colname] == 'False', colname] = '0.0'
            users_df.loc[users_df[colname] == 'FALSE', colname] = '0.0'
            users_df.loc[users_df[colname] == 'True', colname] = '1.0'
            users_df.loc[users_df[colname] == 'TRUE', colname] = '1.0'
        users_df.is_tweeter = pd.to_numeric(users_df.is_tweeter, errors='coerce')
        users_df.fillna({
            'is_tweeter': 0
        }, inplace=True)

        has_location = users_df['location'].notna()
        users_df[f'has_location'] = pd.to_numeric(has_location)
        users_df.drop('location', axis=1, inplace=True)

        convert_str_features_to_numeric(users_df, filename=filename)

        # Fillna bools
        cols_bools = users_df.select_dtypes(include='bool').columns
        users_df.fillna({col: False for col in cols_bools}, inplace=True)
        if users_df.isnull().any().any():
            print(f"\t {filename} has null")
            print(f"\t {users_df.isnull().any()}")
        users_df[cols_bools].astype(int, copy=False)
        users_df = users_df.astype(int, copy=False)

        # All null values should be fixed
        assert not users_df.isnull().any().any()

        # ------------------------------------------
        # Calculate user impact score
        # ------------------------------------------
        users_df[['score']] = np.log(users_df.listed_count + args.theta).add(
            2 * np.log(users_df.friends_count + args.theta)).subtract(np.log(users_df.followers_count + args.theta))
        all_user_feat_d[filename] = users_df

        if args.make_graph:
            assert all_Gu_d is not None
            G_usr_directed = make_Gu(users_df, filename, example_name, labels_d, args, tweet_df)
            if G_usr_directed is not None:
                all_Gu_d[filename] = G_usr_directed
        args.n_processed += 1
    except Exception:
        print(f"\t{filename} Error getting user features")
        traceback.print_exc()


def make_Gu(users_df, filename, example_name, labels_d, args, tweet_df):
    # TODO: The part about constructing edges must be separated from processing users.tsv

    G_usr = nx.from_pandas_edgelist(tweet_df, target='user_id', source='root_user_id')

    try:

        if G_usr.edges == []:
            print(f"\tError {filename}: edge empty")
            return None

        nx.to_directed(G_usr)
        G_usr_directed = nx.DiGraph(label=labels_d[filename])

        impact_score = users_df[['score']]

        edge_index_weight = []

        for edge in G_usr.edges():
            src, tgt = edge
            impact_s_t = impact_t_s = 1.

            # Ignore this part. edge_weight_mode is binary
            # by default, as in the paper
            if args.edge_weight_mode != 'binary' and src != 0:
                try:

                    s_src = impact_score.loc[src].values[0]
                    s_tgt = impact_score.loc[tgt].values[0]
                    impact_s_t, impact_t_s = get_edge_weight(s_src, s_tgt, args)

                except:
                    impact_s_t = impact_t_s = 1.

            edge_index_weight += [(src, tgt, impact_s_t)]
            edge_index_weight += [(tgt, src, impact_t_s)]
        G_usr_directed.add_weighted_edges_from((u, v, w) for u, v, w in edge_index_weight)

        if args.standardize_impact_score:
            # Standardize impact scores
            transformer = StandardScaler().fit(impact_score)
            impact_score_transformed = transformer.transform(impact_score)
        else:
            # No transformations
            impact_score_transformed = impact_score.to_numpy().ravel()
        impact_score_transformed = np.clip(impact_score_transformed, args.min_user_impact, args.max_user_impact)

        impact_score_d = dict(zip(impact_score.index.to_list(), list(impact_score_transformed)))
        impact_score_d = {k: impact_score_d[k] if k in impact_score_d else args.default_user_impact for k in
                          G_usr_directed.nodes}
        nx.set_node_attributes(G_usr_directed, impact_score_d, "score")

        keys = ['followers_count', 'friends_count', 'listed_count', 'favourites_count']
        features_d = users_df[keys].to_dict()
        for feature_name, values_d in features_d.items():
            try:
                nx.set_node_attributes(G_usr_directed, values_d, feature_name)
            except:
                print(f"\t{filename} Error setting attribute {feature_name}")
                traceback.print_exc()
                nx.set_node_attributes(G_usr_directed, dict(zip(list(G_usr_directed), [1] * len(G_usr_directed))),
                                       feature_name)

        return G_usr_directed
    except Exception:
        print(f"\t{filename} Error making graph")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="gossipcop", choices=["politifact", "gossipcop"],
                        help='which dataset to use')
    parser.add_argument('--strategy', type=str, default="copy", choices=["copy", "drop", "random"],
                        help='Use what strategy to fill the missing values')
    parser.add_argument('--outdir', default="outputs", help='path to output directory')
    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--make_graph', default=True, help='Construct Networkx graph?')
    parser.add_argument('--normalize_features', default=True, help='Normalize user features?')
    parser.add_argument('--theta', default=1,
                        help='Smoothing factor for calculating user impact. This ensures that user impact is always >=0')

    parser.add_argument('--epsilon', default=1e-3,
                        help='Smoothing factor to calculate relative user impact. To ensure that user impact is nonzero')

    parser.add_argument('--min_user_impact', type=float, default=1e-2)
    parser.add_argument('--max_user_impact', type=float, default=1e2)
    parser.add_argument('--default_user_impact', type=float, default=1e-2)

    parser.add_argument('--edge_weight_mode', default="binary",
                        choices=["ratio", "reversed_ratio", "product", "binary"],
                        help='For user graph, how do we combine the impact score of two users as the edge weight? Or just binary values in [0, 1]')

    parser.add_argument('--standardize_impact_score', action='store_true',
                        help='Do we standardize impact scores of users?')

    args = parser.parse_args()
    pd.set_option('display.max_columns', 20)

    dataset_names = utils.get_dataset_names(args.dataset)
    #print(dataset_names)
    data_list = utils.get_data_list()
    #print(data_list)

    if not osp.exists(osp.join(utils.get_root_dir(), f"{args.dataset}_labels.pt")):
        print(f"Loading {args.dataset} labels ...")
        examples_real = [(example_real, 0) for example_real in data_list[f'{args.dataset}_real']]
        examples_fake = [(example_fake, 1) for example_fake in data_list[f'{args.dataset}_fake']]
        labels_d = dict(examples_fake + examples_real)
        utils.save_labels(args.dataset, labels_d)
    else:
        print(f"Loading {args.dataset} labels ...")
        labels_d = utils.load_labels(args.dataset)

    if args.debug:

        # Examine specific training examples
        job_list = [["politifact15514", config.POLITIFACT_FAKE_NAME],
                    ["politifact11773", config.POLITIFACT_FAKE_NAME]]

    else:

        # ------------------------------------------
        # job_list was originally for multiprocessing
        # ------------------------------------------
        job_list = []
        # Either X_real or X_fake
        for dataset in dataset_names[args.dataset]:
            for filename in data_list[dataset]:
                job_list += [[filename, dataset]]

    utils.print_heading(args.dataset)
    random.shuffle(job_list)

    print("Loading tweets...")
    all_tweets_d, all_replies_d, all_tweets_score_d = utils.load_tweets(args.dataset)


    all_user_feat_d, all_Gu_d = {}, {}

    args.n_processed = 0

    for [filename, dataset] in job_list:
        if filename not in all_Gu_d.keys():
            process_example(filename, dataset, labels_d=labels_d, all_user_feat_d=all_user_feat_d,
                            all_Gu_d=all_Gu_d, all_tweets_d=all_tweets_d, all_replies_d=all_replies_d,
                            all_tweets_score_d=all_tweets_score_d, args=args)
            if args.n_processed > 0 and args.n_processed % 500 == 0:
                print(f"Processed {args.n_processed} files. Temporarily saving {len(all_Gu_d)} examples ...")
                utils.save_users(args.dataset, all_user_feat_d)
                utils.save_Gu(args.dataset, all_Gu_d)

    print("Saving users and Gu ...")

    utils.save_users(args.dataset, all_user_feat_d)
    utils.save_Gu(args.dataset, all_Gu_d)
    print("Done!")


if __name__ == "__main__":
    main()
