
'''
This is the archived version of Graph Triple construction, in which we use NER instead of LDA in topic extraction
'''
import argparse
import itertools
import multiprocessing as mp
import os
import os.path as osp
import pickle
import time
from functools import partial
from multiprocessing.pool import Pool
from multiprocessing import Manager
from operator import itemgetter

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import random
import textdistance as td
import torch
import lda
import config
import scratch
from models.run_pagerank import get_claim_evidence_pair
from preprocess.data_preparation_user import get_edge_weight
from utils import utils, utils_pagerank

from sklearn.feature_extraction.text import CountVectorizer

from utils.utils_pagerank import get_tweet_vocab


def process_example(filename, label, global_news_article_d, all_tweets_d, all_replies_d, all_tweets_score_d,
                    all_nx_graph_d, all_G_triple, all_key_news_context, nlp, root_dir, args):
    t0 = time.process_time()
    if filename in all_tweets_d:
        tweet_df = all_tweets_d[filename]
        reply_df = all_replies_d[filename]
        tweet_score_ori_d = all_tweets_score_d[filename]
    else:
        print(f"\tSKIP {filename}: no tweet_df")
        return
    print(f"{filename} | T: {len(tweet_df)} R: {len(reply_df)}")
    # TODO: Move to a later point
    # ------------------------------------------
    # Get news article entities
    # ------------------------------------------
    tweet_df = pd.concat([tweet_df, reply_df])

    if len(tweet_df) > 4000:
        # TODO
        print(f"\tSKIP {filename}: TOO LARGE!!")
        return

    ent_news_li = utils_pagerank.get_news_article_ents(filename, nlp, global_news_article_d)

    # ------------------------------------------
    # G_usr: files under FNNUserDataset
    # ------------------------------------------

    if filename in all_nx_graph_d:
        G_usr = all_nx_graph_d[filename]
    else:
        print(f"\tSKIP {filename}: no user graph")
        return
    # Set node types to "Usr"
    nx.set_node_attributes(G_usr, dict(zip(list(G_usr), ['usr'] * len(G_usr))), "type")

    G_usr = utils_pagerank.set_Gu_edge(G_usr, args)

    # Convert tweet id to user id
    twt2usr = dict(
        zip(tweet_df.index.values.tolist(), tweet_df.user_id.values.tolist()))

    lda_model = lda.LDA(n_topics=20, n_iter=1500, random_state=args.seed)

    word_ids_li, vocab = get_tweet_vocab(tweet_df)
    lda_model.fit(word_ids_li)


    # ------------------------------------------
    # G_twt:
    # ------------------------------------------

    G_twt = nx.from_pandas_edgelist(tweet_df.reset_index(), source='root_tweet_id', target='index',
                                    edge_attr=['text', 'index', 'user_id', 'created_at', 'type'])

    nx.set_node_attributes(G_twt, tweet_df.user_id.to_dict(), "user_id")
    nx.set_node_attributes(G_twt, {tweet_id: tweet_id for tweet_id in tweet_df.index}, "tweet_id")

    # Root node id set to -1
    if 0 in G_twt:
        nx.relabel_nodes(G_twt, {
            0: args.default_root_twt_id
        }, copy=False)
    else:
        print(f"\tSKIP {filename}: no root node")
        return

    G_twt = utils_pagerank.set_Gp_score_type(G_twt, tweet_df, tweet_score_ori_d, args, )

    # Todo: larger model!
    tweet_li = tweet_df['text'].to_list()

    tweet_id_li = tweet_df.index.to_list()

    tweet_docs = list(nlp.pipe(tweet_li))

    tweet_doc_vecs = []
    ent_all_li = []

    # Tweet indices that contains keys ()
    # e.g.: 0 ('Occupy Democrats') -> 774381500196200400

    # Which entities appear in the same tweet
    ent_cooc_li = []

    # Which entities appear in which tweet
    ent_occur_li = []

    # TODO: user info with zip()
    assert len(tweet_id_li) == len(tweet_docs)

    for idx_tweet, (tweet_id, tweet_doc) in enumerate(zip(tweet_id_li, tweet_docs)):

        idx = len(ent_all_li)
        ent_twt_li = []
        for idx_ent, ent in enumerate(tweet_doc.ents):
            if len(ent.text) >= 1:
                ent_twt_li += [ent]
                ent_occur_li.append([len(ent_occur_li), tweet_id])
            else:
                print(ent.text)
        ent_cooc_li += list(itertools.combinations(np.arange(idx, len(tweet_doc.ents), 1), 2))
        ent_all_li += ent_twt_li

        tweet_doc_vecs += [tweet_doc.vector]

    # ------------------------------------------
    # G_key: Get keyword sim score
    # ------------------------------------------

    ent_all_text_li = [ent.text for ent in ent_all_li]

    if ent_all_text_li == []:
        print(f"\tSKIP {filename}: no meaningful entities")
        return

    Adj_ent_li = []
    for idx1, ent1 in enumerate(ent_all_text_li):
        Adj_ent_li.append([utils_pagerank.similarity_func(ent1, ent2) for ent2 in ent_all_text_li])
    Adj_ent_sim = np.asmatrix(Adj_ent_li) - np.eye(len(Adj_ent_li))
    Adj_ent_sim[Adj_ent_sim < args.max_ent_sim] = 0

    G_key = nx.from_numpy_matrix(Adj_ent_sim)

    attr_key_d = {idx_key: ent_text for idx_key, ent_text in enumerate(ent_all_text_li)}
    nx.set_node_attributes(G_key, attr_key_d, "text")

    # ------------------------------------------
    # G_key: Keyword deduplication
    # ------------------------------------------

    # Map the uncombined keyword indices to combined indices
    idx_old2new = {}
    connected_components = nx.connected_components(G_key)
    for supernode in connected_components:
        nodes = sorted(list(supernode))
        idx_old2new[nodes[0]] = nodes[0]
        for node in nodes[1:]:
            G_key = nx.contracted_nodes(G_key, nodes[0], node)
            idx_old2new[node] = nodes[0]

    # Set node types to "Key"
    nx.set_node_attributes(G_key, dict(zip(list(G_key), ['key'] * len(G_key))), "type")

    # elif args.key_edge_mode == "cooccur" and ent_cooc_li != []:
    if ent_cooc_li != []:
        # Translate keyword index for the new co-occurrence list
        ent_cooc_li = np.matrix(ent_cooc_li)
        ent_cooc_li = np.vectorize(idx_old2new.get)(ent_cooc_li)
        ent_cooc_li = np.sort(ent_cooc_li, axis=0)
        assert ent_cooc_li.shape[1] == 2

    ent_occur_li = np.matrix(ent_occur_li)
    ent_occur_li[:, 0] = np.vectorize(idx_old2new.get)(ent_occur_li[:, 0])
    ent_occur_li = np.sort(ent_occur_li, axis=0)
    assert ent_occur_li.shape[1] == 2

    ent_tweet_id_set = set([v for _, v in idx_old2new.items()])
    ent_twt_merged_li = itemgetter(*list(ent_tweet_id_set))(ent_all_li)

    # ------------------------------------------
    # Get mapping of keywords to news sentences and to tweets
    # ------------------------------------------

    ent2news = utils_pagerank.get_ent_news_mapping(ent_news_li, ent_twt_merged_li, args)
    ent2tweet_id_li = utils_pagerank.get_ent2tweet_id_li(ent_occur_li, ent_all_text_li)

    # ------------------------------------------
    # Set the occurrences count of keywords
    # ------------------------------------------

    attr_count_G_key = {}

    for node_id in G_key.nodes:
        if 'contraction' in G_key.nodes[node_id]:
            attr_count_G_key[node_id] = len(G_key.nodes[node_id]["contraction"]) + 1
            del G_key.nodes[node_id]['contraction']
        else:
            attr_count_G_key[node_id] = 1

    nx.set_node_attributes(G_key, attr_count_G_key, "score")
    G_key = utils_pagerank.set_key_graph_edge(G_key, ent_cooc_li, args)

    # ------------------------------------------
    # Pagerank - Graph Construction
    # ------------------------------------------

    # Ground-truth tweet id to new tweet index
    mapping_G_twt = {name: j for j, name in enumerate(G_twt.nodes())}

    # Old unclustered entity id to new entity id
    mapping_G_key = {name: len(G_twt) + j for j, name in enumerate(G_key.nodes())}

    # Ground-truth user id to new user index
    mapping_G_usr = {name: len(G_twt) + len(G_key) + j for j, name in enumerate(G_usr.nodes())}

    nx.relabel_nodes(G_twt, mapping_G_twt, copy=False)
    nx.relabel_nodes(G_key, mapping_G_key, copy=False)
    nx.relabel_nodes(G_usr, mapping_G_usr, copy=False)

    # tweet_doc_vecs = np.array(tweet_doc_vecs).T

    G_twin = nx.compose(G_twt, G_key)
    G_twin = G_twin.to_directed()
    G_triple = nx.compose(G_twin, G_usr)

    # ------------------------------------------
    # Add edges between G_usr and G_twt
    # ------------------------------------------

    edge_li_G_usr_G_twt = zip(tweet_df.user_id.values.tolist(),
                              tweet_df.index)

    for src, tgt in edge_li_G_usr_G_twt:
        idx_src, idx_tgt = mapping_G_usr[src], mapping_G_twt[tgt]
        G_triple.add_edge(idx_src, idx_tgt, weight=1)
        G_triple.add_edge(idx_tgt, idx_src, weight=1)

    ###########################################
    # Add edges between G_usr and G_key
    ###########################################

    for ent_id, tweet_id in ent_occur_li.tolist():
        idx_key, idx_usr = mapping_G_key[ent_id], mapping_G_usr[twt2usr[tweet_id]]
        G_triple.add_edge(idx_key, idx_usr, weight=1)
        G_triple.add_edge(idx_usr, idx_key, weight=1)

    if args.draw:
        print(f"Writing {filename}.gexf")
        nx.write_gexf(G_triple, osp.join("outputs", "gexf", f"{filename}.gexf"))
        # TODO
        print(utils_pagerank.get_node_scores(G_triple, nodetype="usr", attr_score='score'))
        if False:
            scratch.draw_pyvis(G_triple, filename)

    ###########################################
    # Saving results
    ###########################################

    all_G_triple[filename] = (G_triple, [mapping_G_twt, mapping_G_key, mapping_G_usr])

    all_key_news_context[filename] = [ent2news, ent2tweet_id_li]
    print(f"\t{time.process_time() - t0} s")
    args.n_processed += 1

    # draw_with_color(G_triple)

    # ------------------------------------------
    # Pagerank - Compute Score
    # ------------------------------------------

    # sns.displot(
    #     scores_df, x="pr", col="type",
    #     facet_kws=dict(margin_titles=True), log_scale=True,
    # )

    # sns.histplot(
    #     df,
    #     x="index",
    #     multiple="stack",
    #     palette="light:m_r",
    #     edgecolor=".3",
    #     linewidth=.5,
    #     log_scale=False,
    # )


    ########################
    # News Article
    ########################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--default_root_twt_id', type=int, default=-1, help='default root tweet node id')
    parser.add_argument('--root', default="data", help='path to store the raw dataset files')
    parser.add_argument('--max_ent_sim', type=float, default=0.75,
                        help='Maximum similarity score computed by SpaCy to consider the two entities as different entities')
    parser.add_argument('--draw', action="store_true", default=False, help='Output G_triple visualization file?')
    parser.add_argument('--dataset', type=str, choices=["politifact", "gossipcop"], default="politifact", help='which dataset to use')
    parser.add_argument('--entity_file', type=str, default="politifact", help='which dataset to use')
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument('--usr_edge_mode', default="sparse", choices=["sparse", "dense"],
                        help='For user graph, how do we determine the existence of edge between two users, either using replying relationships or make the graph fully connected')
    parser.add_argument('--key_edge_mode', default="combo", choices=["cooc", "sim", "combo", "full"],
                        help='For keyword graph, how do we determine the edge weight between two keywords, either using similarity or make the graph fully connected')

    parser.add_argument('--edge_weight_mode', default="binary", choices=["ratio", "reversed_ratio", "product", "binary"],
                        help='For user graph, how do we combine the impact score of two users as the edge weight? Or just binary')

    parser.add_argument('--twt_score_mode', default="log", choices=["original", "log"],
                        help='For initial score of tweets, whether to use log scale')

    parser.add_argument('--epsilon', default=1e-3,
                        help='Smoothing factor to calculate relative user impact. To ensure that user impact is nonzero')

    parser.add_argument('--use_mpc', action='store_true', help='Use multiple processing')
    parser.add_argument('--sample_ratio', type=float, default=None, help='Sample ratio for testing')


    # parser.add_argument('--overwrite', action="store_true", help='Whether to overwrite new_users.tsv')
    args = parser.parse_args()

    print(f"Usr edge mode: {args.usr_edge_mode}")
    print(f"Key edge mode: {args.key_edge_mode}")
    print(f"Twt score mode: {args.twt_score_mode}")

    DATASET_NAMES = utils.get_dataset_names(args.dataset)

    global_news_article_d = {}
    utils.read_news_articles_text(global_news_article_d, args.dataset)
    nlp = spacy.load("en_core_web_md")


    labels_d = utils.load_labels(args.dataset)

    flag = False

    utils.print_heading(args.dataset)
    all_tweets_d, all_replies_d, all_tweets_score_d = utils.read_tweets_and_scores(args.dataset)
    all_nx_graph_d = utils.load_nx_graphs(args.dataset)



    job_list = []

    suffix = f"_SAMPLE{args.sample_ratio}" if args.sample_ratio is not None else ""

    if args.sample_ratio is not None:
        examples_real = [filename for filename in all_nx_graph_d.keys() if labels_d[filename] == 0]
        examples_fake = [filename for filename in all_nx_graph_d.keys() if labels_d[filename] == 1]
        n_samples = int(args.sample_ratio * len(labels_d) * 0.5)


        filenames_sampled = random.sample(examples_real, n_samples) + random.sample(examples_fake, n_samples)

    else:
        filenames_sampled = all_nx_graph_d.keys()
    for filename in filenames_sampled:
        job_list += [[filename, labels_d[filename]]]
    random.shuffle(job_list)

    if args.use_mpc:
        manager = Manager()
        all_G_triple, all_key_news_context = manager.dict(), manager.dict()
        n_cpus = 1 if args.sample_ratio is not None else mp.cpu_count()-1
        pool = Pool(processes=n_cpus)

        print(f"Using MPC with {n_cpus} processes")

        partial_process_example = partial(process_example, global_news_article_d=global_news_article_d,
                                          all_tweets_d=all_tweets_d, all_replies_d=all_replies_d,
                                          all_tweets_score_d=all_tweets_score_d, all_nx_graph_d=all_nx_graph_d,
                                          all_G_triple=all_G_triple, all_key_news_context=all_key_news_context, nlp=nlp,
                                          root_dir=utils.get_root_dir(),
                                          args=args)
        pool.starmap(partial_process_example, job_list)
        all_G_triple = all_G_triple._getvalue()
        all_key_news_context = all_key_news_context._getvalue()

    else:
        path_all_G_triple = os.path.join(utils.get_root_dir(), f"{args.dataset}_G_triple{suffix}.pt")
        path_all_key_news_context = os.path.join(utils.get_root_dir(), f"{args.dataset}_key_news_context{suffix}.pt")
        if osp.exists(path_all_G_triple) and osp.exists(path_all_key_news_context):
            all_G_triple = torch.load(path_all_G_triple)
            all_key_news_context = torch.load(path_all_key_news_context)
            print(f"Loading G_triple and key_news_context with {len(all_G_triple)} examples from cache ...")

        else:
            all_G_triple, all_key_news_context = {}, {}
        print("Do not use MPC")
        args.n_processed = 0
        for [filename, label] in job_list:
            if filename not in all_G_triple.keys():
                process_example(filename, label, global_news_article_d=global_news_article_d, all_tweets_d=all_tweets_d,
                                all_replies_d=all_replies_d, all_tweets_score_d=all_tweets_score_d,
                                all_nx_graph_d=all_nx_graph_d, all_G_triple=all_G_triple,
                                all_key_news_context=all_key_news_context, nlp=nlp, root_dir=utils.get_root_dir(),
                                args=args)
            else:
                print(f"{filename}: using cached")
            if args.n_processed > 1 and args.n_processed % 100 == 0:
                print(f"Processed {args.n_processed} files. Temporarily saving {len(all_G_triple)} examples ...")
                torch.save(all_G_triple, path_all_G_triple)
                torch.save(all_key_news_context, path_all_key_news_context)

    print("Saving...")
    torch.save(all_G_triple, path_all_G_triple)
    torch.save(all_key_news_context, path_all_key_news_context)
    print("Done!")
