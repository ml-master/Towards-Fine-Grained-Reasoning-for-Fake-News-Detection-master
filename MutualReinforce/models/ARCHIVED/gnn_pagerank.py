'''
A simple example to load the adjacency mat of an example

'''
import argparse
import itertools
import os
import os.path as osp
import pickle
import spacy

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import textdistance as td
import torch
from torch_geometric.utils import from_networkx

import config
from models.pagerank import pagerank_numpy
from preprocess.data_preparation_user import get_edge_weight
from scratch import draw_with_color
import seaborn as sns

# from torch_geometric.data import Data
from utils import utils


def cos_similarity_list(m):
    d = m.T @ m
    norm = (m * m).sum(0, keepdims=True) ** .5
    m_norm = m / norm
    sim = m_norm.T @ m_norm
    return sim


def similarity_func(t1, t2):
    return td.jaccard(t1, t2)


def read_file(dataset):
    with open(f"{config.FAKE_NEWS_DATA}\\{dataset}_news_article_evidence.pkl", "rb") as f:
        examples = pickle.load(f)
    return examples


def set_user_graph_edge(G_usr, args):
    edge_li = []

    s_d = nx.get_node_attributes(G_usr, "score")
    s_d[0] = 1

    # Note: here we use 1 as the impact score of root node
    nx.set_node_attributes(G_usr, {0:1}, "score",)
    if args.edge_mode == "dense":

        for src in G_usr.nodes:
            edge_li += [(src, tgt, get_edge_weight(s_d[src], s_d[tgt], args)) for tgt in G_usr.nodes]

    elif args.edge_mode == "sparse":
        edge_li = [(src, tgt, get_edge_weight(s_d[src], s_d[tgt], args)[0] if (src in s_d and tgt in s_d) else 1 ) for src, tgt in G_usr.edges]
    else:
        raise NotImplementedError
    G_usr.remove_edges_from(list(G_usr.edges))
    G_usr.add_weighted_edges_from(edge_li)

    return G_usr

def get_news_article_ents(filename, nlp, global_news_article_d):
    '''

    :param filename:
    :return: entities that co-exists in tweets
    '''

    text = global_news_article_d[filename]
    # Process the text
    doc = nlp(text)

    # Iterate over the entities
    for ent in doc.ents:
        # Print the entity text and label
        print(ent.text, ent.label_)

def process_example(filename, global_news_article_d, args):
    print(filename)

    pd.options.display.max_columns = 20
    pd.set_option('precision', 20)

    # NOTE: reading as int64 is super important
    dtypes = {
        'root_tweet_id': np.int64,
        'tweet_id': np.int64,
        'root_user_id': np.int64,
        'user_id': np.int64,
    }

    # TODO: Move to a later point
    ############################################
    # Get news article entities
    ############################################

    nlp = spacy.load("en_core_web_lg")
    get_news_article_ents(filename, nlp, global_news_article_d)

    tweet_retweet_comment_df = pd.read_csv(
        f"{config.FAKE_NEWS_DATA}\\{config.POLITIFACT_FAKE_NAME}\\{filename}\\tweets_retweets_comments.tsv", sep='\t',
        dtype=dtypes)

    ############################################
    # G_usr: files under FNNUserDataset
    ############################################

    raw_dir = os.path.join(args.root, args.dataset, 'raw')

    labels_d = torch.load(os.path.join(raw_dir, f"{args.dataset}_labels.pt"))

    all_nx_graph_d = torch.load(os.path.join(raw_dir, f"{args.dataset}_nx_graphs{'_DEBUG' if args.debug else ''}.pt"))

    G_usr = all_nx_graph_d[filename]
    # Set node types to "Usr"
    nx.set_node_attributes(G_usr, dict(zip(list(G_usr), ['usr'] * len(G_usr))), "type")

    G_usr = set_user_graph_edge(G_usr, args)

    # Convert tweet id to user id
    twt2usr = dict(
        zip(tweet_retweet_comment_df.tweet_id.values.tolist(), tweet_retweet_comment_df.user_id.values.tolist()))

    ############################################
    # G_twt:
    ############################################

    G_twt = nx.from_pandas_edgelist(tweet_retweet_comment_df, source='root_tweet_id', target='tweet_id',
                                    edge_attr=['text', 'tweet_id', 'user_id', 'created_at', 'type'])

    # Root node id set to -1
    nx.relabel_nodes(G_twt, {
        0: -1
    }, copy=False)

    # Set node type to "twt"
    nx.set_node_attributes(G_twt, dict(zip(list(G_twt), ['twt'] * len(G_twt))), "type")
    # Set node personalization score to 1
    nx.set_node_attributes(G_twt, dict(zip(list(G_twt), [1] * len(G_twt))), "score")

    entities_filename = f"{args.outdir}\\entities_{args.dataset}.pt"
    entities_d = {}
    if not osp.exists(entities_filename):
        print(f"\t Processing {args.dataset} entities")
        examples = read_file(args.dataset)

        for f_name, example in examples.items():
            entities = [evidence[1] for evidence in example[0]]
            entities_d[f_name] = entities

        torch.save(entities_d, entities_filename)
    else:
        entities_d = torch.load(entities_filename)

    ents = entities_d[filename]

    # Todo: larger model!
    nlp = spacy.load("en_core_web_sm")

    tweet_li = tweet_retweet_comment_df['text'].to_list()

    tweet_id_li = tweet_retweet_comment_df['tweet_id'].to_list()

    tweet_docs = list(nlp.pipe(tweet_li))

    tweet_doc_vecs = []
    ent_all_li = []

    # ent_occur_mat = np.zeros((len(tweet_docs), len()), dtype=np.int32)
    adj_li = []

    # G_keyword =
    attr_twt_quality = {}
    attr_key_d = {}

    # Tweet indices that contains keys ()
    # e.g.: 0 ('Occupy Democrats') -> 774381500196200400

    # Which entities appear in the same tweet
    ent_cooc_li = []

    # Which entities appear in which tweet
    ent_occur_li = []

    # Which users' tweet contain this entity
    ent_usr_li = []

    # TODO: user info with zip()
    assert len(tweet_id_li) == len(tweet_docs)

    for idx_tweet, (tweet_id, tweet_doc) in enumerate(zip(tweet_id_li, tweet_docs)):

        ent_li = []

        idx = len(attr_key_d)
        for idx_ent, ent in enumerate(tweet_doc.ents):
            ent_li += [ent.text]
            attr_key_d[idx] = ent.text
            ent_occur_li.append([len(ent_occur_li), tweet_id])
        ent_cooc_li += list(itertools.combinations(np.arange(idx, len(tweet_doc.ents) + 1, 1), 2))
        ent_all_li += ent_li

        attr_twt_quality[tweet_id] = len(ent_li)

        tweet_doc_vecs += [tweet_doc.vector]

    ############################################
    # Keyword deduplication - get sim score
    ############################################

    Adj_ent_li = []
    for idx1, ent1 in enumerate(ent_all_li):
        Adj_ent_li.append([similarity_func(ent1, ent2) for ent2 in ent_all_li])
    Adj_ent_sim = np.asmatrix(Adj_ent_li)
    Adj_ent_sim[Adj_ent_sim < args.max_ent_sim] = 0

    G_key = nx.from_numpy_matrix(Adj_ent_sim)

    nx.set_node_attributes(G_key, attr_key_d, "text")

    # Set node types to "Key"
    nx.set_node_attributes(G_key, dict(zip(list(G_key), ['key'] * len(G_key))), "type")

    ############################################
    # Keyword deduplication - remove duplicates
    ############################################

    # Map the uncombined keyword indices to combined indices
    idx_old2new = {}
    con_comp = nx.connected_components(G_key)
    for supernode in con_comp:
        nodes = sorted(list(supernode))
        idx_old2new[nodes[0]] = nodes[0]
        for node in nodes[1:]:
            G_key = nx.contracted_nodes(G_key, nodes[0], node)
            idx_old2new[node] = nodes[0]

    # Translate keyword index for the new co-occurrence list
    ent_cooc_li = np.matrix(ent_cooc_li)
    ent_cooc_li = np.vectorize(idx_old2new.get)(ent_cooc_li)
    ent_cooc_li = np.sort(ent_cooc_li, axis=0)
    assert ent_cooc_li.shape[1] == 2

    ent_occur_li = np.matrix(ent_occur_li)
    ent_occur_li[:, 0] = np.vectorize(idx_old2new.get)(ent_occur_li[:, 0])
    ent_cooc_li = np.sort(ent_occur_li, axis=0)
    assert ent_cooc_li.shape[1] == 2

    ############################################
    # Set the occurrences count of keywords
    ############################################

    attr_count_G_key = {}

    for node_id in G_key.nodes:
        if 'contraction' in G_key.nodes[node_id]:
            attr_count_G_key[node_id] = len(G_key.nodes[node_id]["contraction"]) + 1
            del G_key.nodes[node_id]['contraction']
        else:
            attr_count_G_key[node_id] = 1

        # Todo: relink these duplicate keys

        # idx_dup_key = list(G_key.nodes[node_id]["contraction"].keys())
    nx.set_node_attributes(G_key, attr_count_G_key, "score")

    ############################################
    # Pagerank - Graph Construction
    ############################################

    # Ground-truth tweet id to new tweet index
    mapping_G_twt = {name: j for j, name in enumerate(G_twt.nodes())}

    # Old unclustered entity id to new entity id
    mapping_G_key = {name: len(G_twt) + j for j, name in enumerate(G_key.nodes())}
    # mapping_G_key = {name: j for j, name in enumerate(G_key.nodes())}

    # Ground-truth user id to new user index
    mapping_G_usr = {name: len(G_twt) + len(G_usr) + j for j, name in enumerate(G_usr.nodes())}
    # mapping_G_usr = {name: j for j, name in enumerate(G_usr.nodes())}

    nx.relabel_nodes(G_twt, mapping_G_twt, copy=False)
    nx.relabel_nodes(G_key, mapping_G_key, copy=False)
    nx.relabel_nodes(G_usr, mapping_G_usr, copy=False)

    tweet_doc_vecs = np.array(tweet_doc_vecs).T

    G_twin = nx.compose(G_twt, G_key)
    G_twin = G_twin.to_directed()
    G_triple = nx.compose(G_twin, G_usr)

    ############################################
    # Add edges between G_usr and G_twt
    ############################################

    edge_li_G_usr_G_twt = zip(tweet_retweet_comment_df.user_id.values.tolist(),
                              tweet_retweet_comment_df.tweet_id.values.tolist())

    for src, tgt in edge_li_G_usr_G_twt:
        idx_src, idx_tgt = mapping_G_usr[src], mapping_G_twt[tgt]
        print(idx_src, idx_tgt)
        G_triple.add_edge(idx_src, idx_tgt, weight=1)
        G_triple.add_edge(idx_tgt, idx_src, weight=1)

    ###########################################
    # Add edges between G_usr and G_key
    ###########################################

    for ent_id, tweet_id in ent_occur_li.tolist():
        idx_key, idx_usr = mapping_G_key[ent_id], mapping_G_usr[twt2usr[tweet_id]]
        print(idx_key, idx_usr)
        G_triple.add_edge(idx_key, idx_usr, weight=1)
        G_triple.add_edge(idx_usr, idx_key, weight=1)

    tweet_retweet_comment_df.user_id.values.tolist()

    ############################################
    # Pagerank - Compute Score
    ############################################

    pr = pagerank_numpy(G_triple)
    print(pr)
    pr_pers = pagerank_numpy(G_triple, do_personalization=True)
    print(pr_pers)
    df = pd.DataFrame.from_dict(pr, orient='index', columns=['pr']).reset_index()
    df.sort_values('pr', ascending=False, inplace=True)
    # node_type_d = nx.get_node_attributes(G_triple, 'type')
    type_df = pd.DataFrame.from_dict(nx.get_node_attributes(G_triple, 'type'), orient='index', columns=['type']).reset_index()
    pr_pers_df = pd.DataFrame.from_dict(pr_pers, orient='index', columns=['pr_personalized']).reset_index()

    scores_df = df.merge(pr_pers_df, on='index').merge(type_df, on='index')

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

    draw_with_color(G_triple, pagerank_score=pr)

    sim = cos_similarity_list(tweet_doc_vecs)

    # Can refer to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    # for comment_doc in comment_docs:
    #     # TODO: get entities in each comment, then calculate overlapping ents
    #     pass

    ########################
    # News Article
    ########################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--root', default="data", help='path to store the raw dataset files')
    parser.add_argument('--max_ent_sim', type=float, default=0.8,
                        help='Maximum similarity score computed by SpaCy to consider the two entities as different entities')
    parser.add_argument('--draw', type=bool, default=False, help='random seed')
    parser.add_argument('--dataset', type=str, default="politifact", help='which dataset to use')
    parser.add_argument('--entity_file', type=str, default="politifact", help='which dataset to use')
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument('--edge_weight_mode', default="reversed_ratio", choices=["ratio", "reversed_ratio", "product"],
                        help='For user graph, how do we combine the impact score of two users as the edge weight')
    parser.add_argument('--edge_mode', default="sparse", choices=["sparse", "dense"],
                        help='For user graph, how do we determine the existence of edge between two users, either using replying relationships or make the graph fully connected')
    parser.add_argument('--epsilon', default=1e-3,
                        help='Smoothing factor to calculate relative user impact. To ensure that user impact is nonzero')

    parser.add_argument('--debug', action='store_true', help='Debug')

    # parser.add_argument('--overwrite', action="store_true", help='Whether to overwrite new_users.tsv')
    args = parser.parse_args()

    DATASET_NAMES = utils.get_dataset_names(args.dataset)

    global_news_article_d = {}
    utils.read_news_articles_text(global_news_article_d, args.dataset)


    data_list = utils.get_data_list()

    for dataset_name, dataset_list in DATASET_NAMES.items():
        examples_real = [(example_real, 0) for example_real in data_list['politifact_real']]
        examples_fake = [(example_fake, 1) for example_fake in data_list['politifact_fake']]
        labels_d = dict(examples_fake + examples_real)
        torch.save(labels_d, os.path.join(config.FAKE_NEWS_DATA,
                                          f"{dataset_name}_labels{'_DEBUG' if args.debug else ''}.pt"))

    for dataset_name, dataset_list in DATASET_NAMES.items():
        print("#" * 30 + f"\n# Processing {dataset_name}\n" + "#" * 30 + "\n")

        for dataset in dataset_list:
            for filename in data_list[dataset]:

                process_example(filename=filename, global_news_article_d=global_news_article_d, args=args)
            # process_example(filename="politifact15246")
