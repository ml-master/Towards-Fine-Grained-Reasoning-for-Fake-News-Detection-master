"""Process Ranking results of Pagerank"""
import time

"""
Assume Know we already have the PageRank results
We generate each claim-evidence pair in the format:
[CLAIM] [ENT] [EVIDENCE]
or just:
[news sentence] [keyword] [tweet1, tweet2 ...]
"""

import argparse
import os
import os.path as osp
from multiprocessing import Manager
import numpy as np
import pandas as pd
import torch
import random

from utils import utils, utils_pagerank
import pagerank

def get_claim_evidence_pair(G_triple, ent2news, ent2tweet_id_li, tweet_df, mode="initial", pr=None):
    # Sort keywords according to PageRank scores

    if mode == "initial":
        scores_G_key_d = utils_pagerank.get_node_scores(G_triple, "key")
    elif mode == "pr":
        assert pr is not None, "Error: must provide pagerank score"
        scores_G_key_d = utils.get_pr_scores(G_triple, pr, nodetype="key")

    # scores_G_twt_d = utils_pagerank.get_node_scores(G_triple, "twt")
    # scores_G_twt_d_li = sorted(scores_G_twt_d.items(), key=lambda x: x[1], reverse=True)

    scores_G_key_d_li = sorted(scores_G_key_d.items(), key=lambda x: x[1], reverse=True)

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

        # TODO: Plain concat or according to score?
        # tweet_scores = None

        # str, str, list of str
        claim_evi_pair = [news_text, key, tweet_text_li]
        claim_evidence_pairs += [claim_evi_pair]
        # scores_initial += [score_initial]
        metadata += [[tweet_ids, user_ids, idx_key]]
    return claim_evidence_pairs, scores_initial, metadata


def run_pagerank(filename, all_claim_evidence_pairs_d, all_pr_scores_d, all_metadata, args):
    print(filename)
    if filename in all_tweets_d:
        tweet_df = all_tweets_d[filename]
        reply_df = all_replies_d[filename]
    tweet_df = pd.concat([tweet_df, reply_df])

    G_triple, _ = all_G_triple[filename]
    ent2news, ent2tweet_id_li = all_key_news_context[filename]

    t0 = time.process_time()
    pr, pr_pers = pagerank.get_pagerank_score(G_triple, filename, draw_gephi=False)
    print(f"\t{filename} {time.process_time() - t0} s")



    claim_evidence_pairs, scores_initial, metadata = get_claim_evidence_pair(G_triple, ent2news, ent2tweet_id_li, tweet_df)
    assert len(claim_evidence_pairs) == len(metadata)
    all_claim_evidence_pairs_d[filename] = claim_evidence_pairs# [ent2news, ent2tweet_id_li]
    # all_scores_d[filename] = scores_initial
    all_pr_scores_d[filename] = [pr, pr_pers]
    all_metadata[filename] = metadata

    args.n_processed += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--root', default="data", help='path to store the raw dataset files')

    parser.add_argument('--dataset', type=str, choices=["politifact", "gossipcop"], default="politifact",
                        help='which dataset to use')
    parser.add_argument('--outdir', default="outputs", help='path to output directory')

    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--get_data', action='store_true',
                        help='After running pagerank, get the ranked input data for KGAT?')
    parser.add_argument('--pagerank_mode', type=str, default="none", choices=["none", "plain", "personalized"],
                        help='Pagerank mode')

    args = parser.parse_args()

    root_dir = utils.get_root_dir()
    manager = Manager()

    all_G_triple = torch.load(os.path.join(root_dir, f"{args.dataset}_G_triple.pt"))
    all_key_news_context = torch.load(os.path.join(root_dir,
                                                   f"{args.dataset}_key_news_context.pt"))

    all_claim_evidence_pairs_d, all_pr_scores_d, all_metadata = {}, {}, {}

    all_tweets_d, all_replies_d, _ = torch.load(osp.join(utils.get_root_dir(),
                                                                          f"{args.dataset}_tweets.pt"))

    args.n_processed = 0
    for filename in all_key_news_context:
        run_pagerank(filename, all_claim_evidence_pairs_d, all_pr_scores_d, all_metadata, args)
    print(f"Saving {args.n_processed} graphs...")

    ######################################
    # Save Claim Evidence Pair
    ######################################

    torch.save(all_claim_evidence_pairs_d,
               os.path.join(root_dir, f"{args.dataset}_claim_evidence_pairs{'_DEBUG' if args.debug else ''}.pt"))
    torch.save(all_metadata,
               os.path.join(root_dir, f"{args.dataset}_claim_evidence_pairs_metadata{'_DEBUG' if args.debug else ''}.pt"))
    torch.save(all_pr_scores_d, os.path.join(root_dir, f"{args.dataset}_pr_scores{'_DEBUG' if args.debug else ''}.pt"))
    print("Done!")
