'''
Genertes G_triple and auxillary information (mappings ...) for calculating pr
'''
import argparse
import configparser
import multiprocessing as mp
import os
import os.path as osp
import random
import time
import traceback
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool

import networkx as nx
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append("..")
from utils import utils, utils_pagerank
from utils.utils import get_processed_dir, get_root_dir
from utils.utils_data import sample, count_real_fake
from utils.utils_pagerank import make_Gk
import config

def process_example(filename, label, global_news_article_evidence_d, all_tweets_d, all_replies_d, all_tweets_score_d,
                    all_Gu,
                    all_G_triple, topic_info_d, args, config):
    t0 = time.process_time()
    # TODO
    # filename = 'gossipcop-6634025418'
    # filename = 'gossipcop-597133430'
    if filename in all_tweets_d:
        tweet_df = all_tweets_d[filename]
        reply_df = all_replies_d[filename]
        tweet_score_ori_d = all_tweets_score_d[filename]
    else:
        print(f"\tSKIP {filename}: no tweet_df")
        return
    print(f"{filename} | T: {len(tweet_df)} R: {len(reply_df)}")

    tweet_df = pd.concat([tweet_df, reply_df])

    if args.case_study:
        tweet_df = pd.read_csv(osp.join("case_study", "case_study_tweets.tsv"), sep="\t", dtype={
            'root_tweet_id': np.int64,
            'tweet_id': np.int64,
            'root_user_id': np.int64,
            'user_id': np.int64
        })
        tweet_df.set_index('tweet_id', inplace=True)
        with open(osp.join("case_study", "case_study_news_article.txt")) as f:
            lines = f.readlines()
            news_article_evidence = ([], lines)
    else:
        if len(tweet_df) > 4000:
            # TODO!!
            print(f"\tSKIP {filename}: TOO LARGE!!")
            return

        # ------------------------------------------
        # Get news article and evidence
        # ------------------------------------------

        news_article_evidence = global_news_article_evidence_d.get(filename, None)

        if news_article_evidence is None:
            print(f"\t{filename} | {label}: empty news sentences")
            return
        elif news_article_evidence[1] == []:
            print(f"\t{filename} | {label}: empty evidence")
            return

        # ------------------------------------------
        # Gu: files under FNNUserDataset
        # ------------------------------------------

        if filename in all_Gu:
            Gu = all_Gu[filename]
        else:
            print(f"\tSKIP {filename} | {label}: no user graph")
            return

    # Set node types to "Usr"
    nx.set_node_attributes(Gu, dict(zip(list(Gu), ['usr'] * len(Gu))), "type")

    Gu = utils_pagerank.reset_Gu_edge_and_root(Gu, args)
    Gu = Gu.to_undirected()

    # ------------------------------------------
    # Gp
    # ------------------------------------------

    tweet_li = tweet_df['text'].to_list()

    # n_p: Number of posts
    n_p = len(tweet_df)
    Adj_Gp = np.zeros((n_p, n_p))
    for i in range(n_p):
        for j in range(n_p):
            if i != j:
                Adj_Gp[i, j] = utils_pagerank.similarity_func(tweet_li[i], tweet_li[j])

    Gp = nx.from_numpy_matrix(Adj_Gp, create_using=nx.Graph)

    # index of the tweet in (0, n_p) -> ground-truth tweet_ids
    idx_p2tweet_id = dict(zip(range(len(tweet_df)), tweet_df.index))

    # ground-truth tweet_ids -> index of the tweet in (0, n_twts)
    mapping_Gp = {tweet_id: idx_twt for idx_twt, tweet_id in idx_p2tweet_id.items()}
    attr_Gp_user_id = dict(zip(range(len(tweet_df)), tweet_df.user_id))

    nx.set_node_attributes(Gp, idx_p2tweet_id, "tweet_id")
    nx.set_node_attributes(Gp, attr_Gp_user_id, "user_id")

    Gp = utils_pagerank.set_Gp_score_type(Gp, idx_p2tweet_id, tweet_score_ori_d, args)

    # ------------------------------------------
    # Gk: Keyword graph
    # ------------------------------------------
    num_topics = config["pagerank"].getint("num_topics", 5)
    num_words = config["pagerank"].getint("num_words_per_topic", 7)
    results = make_Gk(tweet_df, num_topics=num_topics, num_words=num_words, news_article_evidence=news_article_evidence,
                      args=args, config=config)
    if results is None:
        return

    Gk, edges_Gk_Gp, edges_Gk_Gu, corpora_dict, tweet_word_li, weight_mats, topic_vecs = results

    topic_info_d[filename] = (topic_vecs, corpora_dict)

    # ------------------------------------------
    # Pagerank - Graph Construction
    # ------------------------------------------

    # Offset by the number of nodes in Gp
    mapping_Gk = {name: len(Gp) + j for j, name in enumerate(sorted(Gk.nodes))}

    # Ground-truth user id to new user index
    mapping_Gu = {user_id: len(Gp) + len(Gk) + j for j, user_id in enumerate(Gu.nodes())}

    nx.relabel_nodes(Gk, mapping_Gk, copy=False)
    nx.relabel_nodes(Gu, mapping_Gu, copy=False)

    # Set user_id as node attribute
    nx.set_node_attributes(Gu, {v: k for k, v in mapping_Gu.items()}, "user_id")

    G_twin = nx.compose(Gp, Gk)
    G_triple = nx.compose(G_twin, Gu)

    # ------------------------------------------
    # Add edges between Gu and Gp
    # ------------------------------------------

    edges_Gu_Gp = zip(tweet_df.user_id.values.tolist(),
                      tweet_df.index)

    for user_id, tweet_id in edges_Gu_Gp:
        idx_Gu, idx_Gp = mapping_Gu[user_id], mapping_Gp[tweet_id]
        G_triple.add_edge(idx_Gu, idx_Gp, weight=1)

    # ------------------------------------------
    # Add edges between Gu and Gk
    # ------------------------------------------

    for key_id, user_id in edges_Gk_Gu:
        try:
            idx_Gk, idx_Gu = mapping_Gk[key_id], mapping_Gu[user_id]
            G_triple.add_edge(idx_Gk, idx_Gu, weight=1)
        except:
            print(f"\t{filename} Error adding edges between Gk and Gu")
            traceback.print_exc()

    # ------------------------------------------
    # Add edges between Gp and Gk
    # ------------------------------------------

    for key_id, tweet_id in edges_Gk_Gp:
        idx_Gk, idx_Gp = mapping_Gk[key_id], mapping_Gp[tweet_id]
        G_triple.add_edge(idx_Gk, idx_Gp, weight=1)

    if args.draw:
        print(f"Writing {filename}.gexf")
        nx.write_gexf(G_triple, osp.join("outputs", "gexf", f"{filename}.gexf"))

        # Another thing we can use is pyvis
        # This drawing tool is interactive
        # scratch.draw_pyvis(G_triple, filename)

    # ------------------------------------------
    # Saving results
    # ------------------------------------------

    mappings = [mapping_Gp, mapping_Gk, mapping_Gu]

    all_G_triple[filename] = (G_triple, mappings, weight_mats)

    print(f"\t{time.process_time() - t0} s")
    args.n_processed += 1

    # draw_with_color(G_triple)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--default_root_twt_id', type=int, default=-1, help='default root tweet node id')
    parser.add_argument('--default_user_impact', type=float, default=1e-2)
    parser.add_argument('--root', default="data", help='path to store the raw dataset files')
    parser.add_argument('--draw', action="store_true", default=False, help='Output G_triple visualization file?')
    parser.add_argument('--dataset', type=str, choices=["politifact", "gossipcop"], default="politifact",
                        help='which dasataset to use')
    parser.add_argument('--outdir', required=True, help='path to output directory')

    parser.add_argument('--edge_weight_mode', default="binary",
                        choices=["ratio", "reversed_ratio", "product", "binary"],
                        help='For user graph, how do we combine the impact score of two users as the edge weight? Or just binary')

    parser.add_argument('--twt_score_mode', default="log", choices=["original", "log"],
                        help='For personalization score of tweets, whether to use log scale')

    parser.add_argument('--key_score_mode', default="log", choices=["original", "log"],
                              help='For personalization score of keywords, whether to use log scale')

    parser.add_argument('--epsilon', default=1e-3,
                        help='Smoothing factor to calculate relative user impact. To ensure that user impact is nonzero')

    parser.add_argument('--config_file', type=str, required=True)

    parser.add_argument('--use_mpc', action='store_true', help='Use multiple processing')
    parser.add_argument('--reprocess_input', action='store_true', default=False,
                        help='whether to redo the processing of input')

    parser.add_argument("--case_study", action='store_true', default=False,
                        help='Case study for paper')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = configparser.ConfigParser()
    path_config_file = osp.join(args.config_file)
    if osp.exists(path_config_file):
        print(f"Loading {args.config_file}")
        config.read(path_config_file)
    else:
        raise Exception("Cannot find config file")
    args.dataset = config["KGAT"].get("dataset")

    if not osp.exists(args.outdir):
        os.mkdir(args.outdir)

    DATASET_NAMES = utils.get_dataset_names(args.dataset)

    global_news_article_evidence_d = utils.read_news_article_evidence(args.dataset)
    labels_d = utils.load_labels(args.dataset)

    utils.print_heading(args.dataset)
    all_tweets_d, all_replies_d, all_tweets_score_d = utils.read_tweets_and_scores(args.dataset)
    all_Gu = utils.load_nx_graphs(args.dataset)
    topic_info_d = {}

    job_list = []
    sample_ratio = config["KGAT"].getfloat("sample_ratio", None)
    balanced = config["KGAT"].getboolean("balanced", False)
    sample_suffix = f"_SAMPLE{sample_ratio}" if sample_ratio is not None else ""
    sample_suffix += "_bal" if balanced else ""
    # exp_name = f"{exp_name}{sample_ratio if sample_ratio is not None else ''}"

    processed_dir = get_root_dir()

    if sample_ratio is not None:
        # filenames_sampled = sample(all_Gu.keys(), labels_d, sample_ratio, balanced)
        filenames_sampled = sample(global_news_article_evidence_d.keys(), labels_d, sample_ratio, balanced)

    elif args.case_study:
        filenames_sampled = ["politifact14258"]
        sample_suffix = "case_study"

    else:
        filenames_sampled = list(global_news_article_evidence_d.keys())

    print(f"Sampled {len(filenames_sampled)} examples")
    torch.save(filenames_sampled, osp.join(processed_dir, f"{args.dataset}_filenames{sample_suffix}.pt"))

    n_real, n_fake = count_real_fake(filenames_sampled, labels_d)
    for filename in filenames_sampled:
        job_list += [[filename, labels_d[filename]]]
    random.shuffle(job_list)

    path_all_G_triple = os.path.join(processed_dir, f"{args.dataset}_G_triple.pt")

    path_topic_info_d = os.path.join(processed_dir, f"{args.dataset}_topic_info_d.pt")

    if osp.exists(path_all_G_triple) and not args.reprocess_input:
        all_G_triple = torch.load(path_all_G_triple)

        print(f"Loading G_triple with {len(all_G_triple)} examples from cache ...")

    else:
        all_G_triple = {}
    print("Do not use MPC")
    args.n_processed = 0
    for [filename, label] in job_list:
        if filename not in all_G_triple.keys():
            process_example(filename, label, global_news_article_evidence_d=global_news_article_evidence_d,
                            all_tweets_d=all_tweets_d, all_replies_d=all_replies_d,
                            all_tweets_score_d=all_tweets_score_d,
                            all_Gu=all_Gu, all_G_triple=all_G_triple, topic_info_d=topic_info_d, args=args,
                            config=config)
        else:
            print(f"{filename}: using cached")
        if args.n_processed > 1 and args.n_processed % 200 == 0:
            print(f"Processed {args.n_processed} files. Temporarily saving {len(all_G_triple)} examples ...")
            torch.save(all_G_triple, path_all_G_triple)
            torch.save(topic_info_d, path_topic_info_d)

    print("Saving...")
    torch.save(all_G_triple, path_all_G_triple)
    torch.save(topic_info_d, path_topic_info_d)
    count_real_fake(all_G_triple, labels_d)
    count_real_fake(topic_info_d, labels_d)
    print("Done!")
