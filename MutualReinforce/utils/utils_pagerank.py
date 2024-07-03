import itertools
import pickle

import gensim
import networkx as nx
import numpy as np
import textdistance as td
from gensim.models import ldamodel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

import config
from models.get_Gu import get_edge_weight
from utils.utils_data import get_sent_topic_weight_dist, get_pr_scores_of_nodetype


def cos_similarity_list(m):
    norm = (m * m).sum(0, keepdims=True) ** .5
    m_norm = m / norm
    sim = m_norm.T @ m_norm
    return sim


def similarity_func(t1, t2):
    sim = td.cosine(t1, t2)
    return abs(sim)


def read_file(dataset):
    with open(f"{config.FAKE_NEWS_DATA}\\{dataset}_news_article_evidence.pkl", "rb") as f:
        examples = pickle.load(f)
    return examples


def get_ent_news_mapping(ent_news_li, ent_twt_li, args):
    '''
    Get the context of the given entities within the news sentences
    :param ent_news_li:
    :param ent_twt_li:
    :param args:
    :return ent2news: mapping from entity text to news sentences
        dict: str -> str
    '''
    ent2news = {}
    for ent_twt in ent_twt_li:
        for ent_news in ent_news_li:
            # if ent_twt.similarity(ent_news) > args.max_ent_sim:
            if similarity_func(ent_twt.text, ent_news.text) > args.max_ent_sim:
                if ent_twt.text not in ent2news:
                    ent2news[ent_twt.text] = [ent_news.sent.text]
                else:
                    ent2news[ent_twt.text] += [ent_news.sent.text]
    return ent2news


def get_Gk_score(Gk):
    key_d = nx.get_node_attributes(Gk, 'text')
    score_d = nx.get_node_attributes(Gk, 'score')
    key2score = dict(zip(list(key_d.values()), list(score_d.values())))
    return key2score


def get_node_scores(G_triple, nodetype=None, attr_score='score', normalized=False):
    '''

    :param G_triple:
    :param nodetype:
    :param attr_score: Whether to return the initial score or final score of PageRank.
            Can be either 'score' or 'final_score'
    :return: dictionary of scores
    '''

    score_d = nx.get_node_attributes(G_triple, attr_score)
    denom = sum(score_d.values()) if normalized else 1

    if nodetype is not None:
        node_ids = [node for node in G_triple.nodes if G_triple.nodes[node]['type'] == nodetype]
        score_d = {k: score_d[k] / denom for k in node_ids}
    return score_d


def get_news_article_ents(filename, nlp, global_news_article_d):
    '''

    :param filename:
    :return: entities that co-exists in tweets
    '''
    ents_keep = {'PERSON', 'NORP', 'ORG', 'GPE'}

    text = global_news_article_d[filename]
    # Process the text
    doc = nlp(text)

    ent_li = [ent for ent in doc.ents if ent.label_ in ents_keep]

    return ent_li


def get_ent2tweet_id_li(ent_occur_li, ent_all_text_li):
    """
    Map the entity / keyword (str) to corresponding tweet_id list

    :param ent_occur_li: Which tweets does each entity appear in?
    :return: ent2tweet_id_li
    """
    ent2tweet_id_li = {}
    for idx_key, tweet_id in ent_occur_li.tolist():
        key = ent_all_text_li[idx_key]
        if idx_key in ent2tweet_id_li:
            ent2tweet_id_li[key] += [tweet_id]
        else:
            ent2tweet_id_li[key] = [tweet_id]
    return ent2tweet_id_li


# Archived
def set_key_graph_edge(Gk, ent_cooc_li, args):
    '''
    Archived Function. Set the edges for each keyword according to their similarity or co-occurrence
    :param Gk:
    :param ent_cooc_li:
    :param args:
    :return:
    '''
    edges_Gk_li = []
    if args.Gk_edge_mode == "cooc" and ent_cooc_li != []:
        for [node1, node2] in ent_cooc_li.tolist():
            if node1 != node2:
                edges_Gk_li += [(node1, node2, 1)]

    elif args.Gk_edge_mode in ["sim", "combo"]:
        nodes = Gk.nodes
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    edges_Gk_li += [(node1, node2, similarity_func(nodes[node1]['text'], nodes[node2]['text']))]

    elif args.Gk_edge_mode == "full":
        # Fully connected
        nodes = Gk.nodes
        for node1 in nodes:
            edges_Gk_li += [(node1, node2, 1) for node2 in nodes if node1 != node2]
    else:
        raise NotImplementedError

    Gk.add_weighted_edges_from(edges_Gk_li)
    return Gk


def set_Gu_edge(Gu, args):
    edge_li = []

    s_d = nx.get_node_attributes(Gu, "score")
    s_d[0] = 1

    nx.set_node_attributes(Gu, {
        0: 1
    }, "score", )
    if args.Gu_edge_mode == "dense":

        for src in Gu.nodes:
            edge_li += [(src, tgt, get_edge_weight(s_d[src], s_d[tgt], args)) for tgt in Gu.nodes]

    elif args.Gu_edge_mode == "sparse":
        edge_li = [(src, tgt, get_edge_weight(s_d[src], s_d[tgt], args)[0] if (src in s_d and tgt in s_d) else 1) for
                   src, tgt in Gu.edges]
    else:
        raise NotImplementedError
    Gu.remove_edges_from(list(Gu.edges))
    Gu.add_weighted_edges_from(edge_li)

    return Gu


def set_Gp_score_type(Gp, idx_Gp2tweet_id, tweet_score_ori_d, args):
    '''
    Set G_twt node types and personalization scores
    :param Gp:
    :param mapping_G_twt:
    :param tweet_score_ori_d:
    :param args:
    :return:
    '''

    if args.twt_score_mode == "log":

        tweet_score_d = {k: (np.log(tweet_score_ori_d[idx_Gp2tweet_id[k]] + 1) if idx_Gp2tweet_id[
                                                                                       k] in tweet_score_ori_d else 0) + 1
                         for k in
                         Gp.nodes}

    elif args.twt_score_mode == "original":
        tweet_score_d = {
            k: (tweet_score_ori_d[idx_Gp2tweet_id[k]] if idx_Gp2tweet_id[k] in tweet_score_ori_d else 0) + 1 for k in
            Gp.nodes}
    else:
        raise NotImplementedError

    nx.set_node_attributes(Gp, tweet_score_d, "score")

    # Set node type to "twt"
    nx.set_node_attributes(Gp, dict(zip(list(Gp), ['twt'] * len(Gp))), "type")
    return Gp


# ------------------------------------------
# Below are new implementations based on LDA
# ------------------------------------------


def tokenize_word_list(tokenizer, sent_li, stop):
    sent_word_li = []
    for sent in sent_li:
        sent_word = [word for word in tokenizer.tokenize(sent.lower()) if word not in stop]
        sent_word_li += [sent_word]
    return sent_word_li


def make_Gk(tweet_df, num_topics=20, num_words=5, news_article_evidence=None, args=None, config=None):
    stop = set(stopwords.words('english'))
    stop.update(['href', 'br', 'via', 'url'])

    tokenizer = RegexpTokenizer(r'\w+')

    # Store the lower-cased tweets as a list
    tweet_li = tweet_df.text.str.lower().to_list()

    # Store each tweet as a list of words
    tweet_word_li = tokenize_word_list(tokenizer, tweet_li, stop)

    news_article_sent_li = news_article_evidence[1]
    news_article_evi_li = [evi[2] for evi in news_article_evidence[0]]

    sent_word_li = tokenize_word_list(tokenizer, news_article_sent_li, stop)
    evi_word_li = tokenize_word_list(tokenizer, news_article_evi_li, stop)

    # The corpora is constructed through the joint word sets
    corpora_dict = gensim.corpora.Dictionary(tweet_word_li + sent_word_li + evi_word_li)
    assert tweet_df.index.min() > max(corpora_dict.keys())

    bow_sents = [corpora_dict.doc2bow(sent_words) for sent_words in sent_word_li]
    bow_evis = [corpora_dict.doc2bow(evi_words) for evi_words in evi_word_li]
    bow_tweets = [corpora_dict.doc2bow(tweet_words) for tweet_words in tweet_word_li]

    # bows = (bow_sents, bow_evis, bow_tweets)

    lda = ldamodel.LdaModel(corpus=bow_tweets + bow_sents, id2word=corpora_dict, num_topics=num_topics)

    # ------------------------------------------
    # Get topic sim mat by their representation vecs
    # ------------------------------------------

    # num_topics * num_topics
    topic_vecs = lda.get_topics()

    # ------------------------------------------
    # Get mapping of keywords to news sentences and to tweets
    # ------------------------------------------

    # Which words in Gk are most related to a topic?
    # # i.e., word distribution of each topic
    # topic_words_weight_d: topic_id -> dict of weight of each word under this topic
    # num_topics * num_nodes_in_G_key
    word2topic_weight_mat, nodes_G_key, edges_G_key = get_word2topic_weight_matrix(lda, num_topics, num_words,
                                                                                   corpora_dict)
    assert word2topic_weight_mat.shape == (num_topics, len(corpora_dict))

    # How important is each of the topics to a tweet / sentence?
    # i.e., topic distribution of each sentence
    # num_sents * num_topics
    topic_dist_mat_sent = get_sent_topic_weight_dist(lda, bow_sents, num_topics, len(news_article_sent_li))
    topic_dist_mat_tweet = get_sent_topic_weight_dist(lda, bow_tweets, num_topics, len(tweet_df))

    # num_sents * num_nodes_in_G_key
    word2sent_weight_mat_sent = topic_dist_mat_sent.dot(word2topic_weight_mat)
    word2sent_weight_mat_tweet = topic_dist_mat_tweet.dot(word2topic_weight_mat)

    # TODO!!: For some news sentences, there are no evidences
    if bow_evis != []:
        topic_dist_mat_evi = get_sent_topic_weight_dist(lda, bow_evis, num_topics)
        word2sent_weight_mat_evi = topic_dist_mat_evi.dot(word2topic_weight_mat)
    else:
        word2sent_weight_mat_evi = None

    weight_mats = (word2topic_weight_mat, topic_dist_mat_sent, topic_dist_mat_tweet)

    assert word2sent_weight_mat_sent.shape[1] == len(corpora_dict)

    # ------------------------------------------
    # Gk Graph Construction
    # ------------------------------------------

    edges_Gk_Gp = []
    edges_Gk_Gu = []

    Gk = nx.from_edgelist(edges_G_key, create_using=nx.Graph)
    assert len(Gk.nodes) == len(nodes_G_key)

    # Link each tweet to the keywords it mentions (nodes in Gk)
    for i in range(len(tweet_df)):

        # Get intersection between words mentioned in tweet and nodes_G_key
        tweet_word_ids = [corpus_word_tup[0] for corpus_word_tup in bow_tweets[i]]
        tweet_key_ids = set(tweet_word_ids) & set(Gk.nodes)

        for key_id in tweet_key_ids:

            # Edge weight between Gk and Gp are all 1
            tweet_id = tweet_df.index[i]
            user_id = tweet_df.user_id[tweet_id]
            # One user has multiple tweets
            if not isinstance(user_id, np.int64) and not isinstance(user_id, np.float64):
                user_id = user_id.iloc[0]
            edges_Gk_Gp += [[key_id, tweet_id]]
            edges_Gk_Gu += [[key_id, user_id]]

    if args.key_score_mode == "log":
        attr_key_score = {k: (np.log(corpora_dict.dfs[k] + 1) + 1) for k in
                          Gk.nodes}
    elif args.key_score_mode == "original":
        attr_key_score = corpora_dict.dfs
    else:
        raise NotImplementedError

    # Set node types to "Key"
    nx.set_node_attributes(Gk, dict(zip(list(Gk), ['key'] * len(Gk))), "type")
    nx.set_node_attributes(Gk, attr_key_score, "score")
    nx.set_node_attributes(Gk, corpora_dict.id2token, "text")

    results = (Gk, edges_Gk_Gp, edges_Gk_Gu, corpora_dict, tweet_word_li, weight_mats, topic_vecs)

    return results


def get_word2topic_weight_matrix(model, num_topics=20, num_words=5, corpora_dict=None):
    topic_scores = model.show_topics(num_topics, num_words, formatted=False)
    topic_idx_li = []
    word_idx_li = []
    weight_li = []

    # Union all keywords in all topics covered by all the tweets
    nodes_Gk, edges_Gk = set(), set()
    for topic_id, word_weight_li in topic_scores:
        topic_idx_li += [topic_id] * len(word_weight_li)
        topic_word_ids_li = [corpora_dict.token2id[x[0]] for x in word_weight_li]
        nodes_Gk.update(topic_word_ids_li)
        edges_Gk.update(list(itertools.combinations(topic_word_ids_li, 2)))
        for (word, weight) in word_weight_li:
            word_idx_li += [corpora_dict.token2id[word]]
            weight_li += [weight]  # data

    word2topic = coo_matrix((weight_li, (topic_idx_li, word_idx_li)), shape=(num_topics, len(corpora_dict)))
    edges_Gk = sorted(list(edges_Gk), key=lambda x: (x[0], x[1]))

    return word2topic, list(nodes_Gk), list(edges_Gk)


# Archived
def get_tweet_vocab(tweet_df):
    '''
    This function is archived. We now use gensim.models.ldamodel
    :param tweet_df:
    :return:
    '''
    tweet_li = tweet_df.text.str.lower()
    cv = CountVectorizer()
    cv.fit_transform(tweet_li)
    vocab = cv.vocabulary_.copy()

    word_ids_li = []

    for tweet in tweet_li:
        word_ids_li += [[vocab[w] for w in tweet.split() if w in vocab]]

    return word_ids_li, vocab


def reset_Gu_edge_and_root(Gu, args, reset_edge_weight=True):
    # impact score of root node = 1
    nx.set_node_attributes(Gu, {
        0: args.default_user_impact
    }, "score")
    if reset_edge_weight:
        for u, v, d in Gu.edges(data=True):
            d['weight'] = 1.

    return Gu


def fit_Gk_score_sparse_vector(G_triple, pr, idx_Gk2bow_id, corpora_dict):
    """
    Conver scores for each bow to scores of nodes in Gk
    :param G_triple:
    :param pr:
    :param idx_Gk2bow_id:
    :param corpora_dict:
    :return:
    """
    scores_Gk_d = get_pr_scores_of_nodetype(G_triple, pr, nodetype="key")
    score_li, bow_id_li = [], []
    for idx_key, score in scores_Gk_d.items():
        bow_id_li += [idx_Gk2bow_id[idx_key]]
        score_li += [score]
    Gk_score_vec = coo_matrix((score_li, ([0] * len(score_li), bow_id_li)), shape=(1, len(corpora_dict)))
    return Gk_score_vec.T
