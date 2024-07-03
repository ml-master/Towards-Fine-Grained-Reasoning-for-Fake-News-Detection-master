"""Process Ranking results of Pagerank"""
import argparse
import configparser
import os
import os.path as osp
import random

import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd
import torch

import pagerank
from models.agg_node_embedding import Net, get_user_embeddings
from utils import utils, utils_pagerank
from utils import utils_data
from utils.utils import get_root_dir
from utils.utils_data import get_pr_score_vecs, count_real_fake, select_claim_evi_pairs
from utils.utils_pagerank import fit_Gk_score_sparse_vector


def get_claim_evidence_pair(G_triple, ent2news, ent2tweet_id_li, tweet_df, mode="initial", pr=None) -> object:
    # Sort keywords according to PageRank scores
    if mode == "initial":
        scores_Gk_d = utils_pagerank.get_node_scores(G_triple, "key")
    elif mode == "pr":
        assert pr is not None, "Error: must provide pagerank score"
        scores_Gk_d = utils_data.get_pr_scores_of_nodetype(G_triple, pr, nodetype="key")

    scores_G_key_d_li = sorted(scores_Gk_d.items(), key=lambda x: x[1], reverse=True)

    claim_evidence_pairs = []

    # Meta info includes the idx of keyword / tweet_id / user_id in each pair, as well as the initial score
    scores_initial, metadata = [], []

    # For each entity mentioned in tweets (sorted by score)
    for idx_key, score_initial in scores_G_key_d_li:
        key = G_triple.nodes[idx_key]['text']
        if key in ent2news:
            news_text = ent2news[key][0]
        else:
            news_text = ""
        tweet_ids = ent2tweet_id_li[key]

        tweet_text_li = tweet_df.loc[tweet_ids].text.to_list()

        user_ids = (tweet_df.loc[tweet_ids].user_id).to_list()

        # str, str, list of str
        claim_evi_pair = [news_text, key, tweet_text_li]
        claim_evidence_pairs += [claim_evi_pair]
        # scores_initial += [score_initial]
        metadata += [[tweet_ids, user_ids, idx_key]]
    return claim_evidence_pairs, scores_initial, metadata


def run(filename, all_tweets_d, all_replies_d, all_G_triple, topic_info_d, global_news_article_evidence_d, all_mr_d,
        all_claim_evi_pairs_d, all_user_embed_d, model, labels_d, stats_d, args, config):
    # filename = "gossipcop-1077224233"
    if filename in all_tweets_d:
        tweet_df = all_tweets_d[filename]
        reply_df = all_replies_d[filename]
    else:
        print(f"\tSKIP {filename}: no tweet_df")
        return
    print(f"{filename} | T: {len(tweet_df)} R: {len(reply_df)}")

    stats = {
        "T": len(tweet_df),
        "R": len(reply_df)
    }

    tweet_df = pd.concat([tweet_df, reply_df])

    news_article_evidence = global_news_article_evidence_d[filename]

    if filename not in all_G_triple:
        if not tweet_df.empty and news_article_evidence[1] != []:
            R_scores, claim_evi_pairs = select_claim_evi_pairs(tweet_df, news_article_evidence[1], config)
            all_mr_d[filename] = [R_scores, None, None, claim_evi_pairs, None]
            all_claim_evi_pairs_d[filename] = claim_evi_pairs
            return
        else:
            return

    G_triple, mappings, weight_mats = all_G_triple[filename]

    pr, pr_pers, runtime = pagerank.get_pagerank_score(G_triple, filename, draw_gephi=False)
    print(f"\tlabel: {labels_d[filename]} | {runtime} s")

    stats["time"] = runtime
    stats_d[filename] = stats

    mapping_Gp, mapping_Gk, mapping_Gu = mappings
    idx_Gk2bow_id = {v: k for k, v in mapping_Gk.items()}
    topic_vecs, corpora_dict = topic_info_d[filename]

    if isinstance(topic_vecs, sp.sparse.csr_matrix):
        topic_vecs = topic_vecs.todense()

    # ------------------------------------------
    # Get news article and evidence
    # ------------------------------------------


    tweet_id2user_id = tweet_df.user_id.to_dict()
    idx_Gp2idx_Gu = {mapping_Gp[k]: mapping_Gu[v] for k, v in tweet_id2user_id.items()}

    Gk_score_vec = fit_Gk_score_sparse_vector(G_triple, pr_pers, idx_Gk2bow_id, corpora_dict)

    R_scores, sim_topics, candidate_topics, twt_weight_mat, claim_evi_pairs, (
        related_idx_usr_all, related_idx_twt_all) = get_pr_score_vecs(weight_mats, G_triple, topic_vecs, pr,
                                                                      idx_Gp2idx_Gu, news_article_evidence, tweet_df,
                                                                      Gk_score_vec, idx_Gk2bow_id, labels_d, config)

    # Map ids in Gu to actual user ids
    idx_Gu2user_id = dict([(v, k) for k, v in mapping_Gu.items()])

    related_user_ids_all = []

    for related_idx_usr_li in related_idx_usr_all:
        related_user_ids_li = [idx_Gu2user_id[idx_usr] for idx_usr in related_idx_usr_li]
        related_user_ids_all += [related_user_ids_li]

    # user_adjacency_nodes = get_adjacent_nodes_of_user(candidate_topics, G_triple, mapping_Gu, related_idx_usr_all)

    scores_G_triple_d = utils_pagerank.get_node_scores(G_triple, normalized=True)

    scores = (scores_G_triple_d, pr, pr_pers)

    all_mr_d[filename] = [R_scores, sim_topics, twt_weight_mat, claim_evi_pairs, scores]

    all_claim_evi_pairs_d[filename] = claim_evi_pairs

    if args.get_user_embeddings:
        user_embed = get_user_embeddings(filename, all_G_triple, args, model)
        all_user_embed_d[filename] = user_embed

    # For each topic
    # topic_dist_mat_tweet: # num_tweets * num_topics
    def draw_gexf_with_attr(G_triple, attr, reset_label=False):

        G_triple_drawing = G_triple.copy()
        nx.set_node_attributes(G_triple_drawing, attr, "init")
        nx.set_node_attributes(G_triple_drawing, pr, "pr")
        nx.set_node_attributes(G_triple_drawing, pr_pers, "pr_pers")
        if reset_label:
            attr_key_text = nx.get_node_attributes(G_triple_drawing, "text")
            nx.set_node_attributes(G_triple_drawing, dict(zip(G_triple.nodes, [" "] * len(G_triple))), "Label")
            nx.set_node_attributes(G_triple_drawing, attr_key_text, "Label")

        print(f"Writing {filename}.gexf")
        nx.write_gexf(G_triple_drawing, osp.join("outputs", "gexf", f"pr_{filename}.gexf"))

    if args.draw:
        draw_gexf_with_attr(G_triple, scores_G_triple_d, True)
        # draw_gexf_with_attr(G_triple, pr, "pr")
        # draw_gexf_with_attr(G_triple, pr_pers, "pr_pers")
    args.n_processed += 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--root', default="data", help='path to store the raw dataset files')
    parser.add_argument('--draw', action="store_true", default=False, help='Output G_triple visualization file?')

    parser.add_argument('--outdir', default="outputs", help='path to output directory')

    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--get_data', action='store_true',
                        help='After running pagerank, get the ranked input data for KGAT?')
    parser.add_argument('--pagerank_mode', type=str, default="none", choices=["none", "plain", "personalized"],
                        help='Pagerank mode')
    parser.add_argument('--config_file', type=str, required=True)

    # ------------------------------------------
    # Arguments for generating user embeddings
    # ------------------------------------------

    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--get_user_embeddings', action='store_true')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--in_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Note that config file is loaded from the FinerFact directory
    config = configparser.ConfigParser()
    path_config_file = osp.join("..", "FineReasoning", "kgat", "config", args.config_file)
    if osp.exists(path_config_file):
        print(f"Loading {args.config_file}")
        config.read(path_config_file)
    else:
        raise Exception("Cannot find config file")
    args.dataset = config["KGAT"].get("dataset")
    args.user_embed_dim = config["KGAT"].getint("user_embed_dim")
    args.dataset = config["KGAT"].get("dataset")
    args.user_embed_dim = config["KGAT"].getint("user_embed_dim")

    # format: claim_evi_pair, news_article, label
    global_news_article_evidence_d = utils.read_news_article_evidence(args.dataset)
    labels_d = utils.load_labels(args.dataset)

    processed_dir = get_root_dir()
    topic_info_d = torch.load(os.path.join(processed_dir, f"{args.dataset}_topic_info_d.pt"))
    all_G_triple = torch.load(os.path.join(processed_dir, f"{args.dataset}_G_triple.pt"))

    model = Net(args)

    all_mr_d, all_claim_evi_pairs_d, all_user_embed_d, stats_d = {}, {}, {}, {}

    count_real_fake(all_G_triple.keys(), labels_d)
    count_real_fake(topic_info_d.keys(), labels_d)

    all_tweets_d, all_replies_d, _ = torch.load(osp.join(processed_dir,
                                                         f"{args.dataset}_tweets.pt"))

    args.n_processed = 0
    for filename in global_news_article_evidence_d:
        run(filename, all_tweets_d, all_replies_d, all_G_triple, topic_info_d, global_news_article_evidence_d, all_mr_d,
            all_claim_evi_pairs_d, all_user_embed_d, model, labels_d, stats_d, args, config)

        if args.n_processed >= 1 and args.n_processed % 100 == 0:
            print(f"Processed {args.n_processed} files. Temporarily saving {len(all_mr_d)} examples ...")
            torch.save(all_mr_d, os.path.join(processed_dir, f"{args.dataset}_all_mr_d.pt"))
            stats_df = pd.DataFrame.from_dict(stats_d).transpose()
            stats_df.to_csv(osp.join(processed_dir, f"{args.dataset}_stats_d.tsv"), sep="\t")
            if all_user_embed_d != {}:
                torch.save(all_user_embed_d,
                           osp.join(processed_dir, f"{args.dataset}_user_embed.pt"))

    # ------------------------------------------
    # Save Claim Evidence Pair
    # ------------------------------------------

    print(f"Saving {args.n_processed} examples...")
    torch.save(all_mr_d, os.path.join(processed_dir, f"{args.dataset}_all_mr_d.pt"))
    if all_user_embed_d != {}:
        torch.save(all_user_embed_d, osp.join(processed_dir, f"{args.dataset}_user_embed.pt"))
    print("Done!")


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', 'stats.txt')
    main()
