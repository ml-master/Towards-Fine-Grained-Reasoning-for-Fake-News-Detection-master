import random

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


def count_real_fake(filenames, labels_d):
    n_real = len([x for x in list(filenames) if labels_d[x] == 0])

    n_fake = len([x for x in list(filenames) if labels_d[x] == 1])
    print(f"Real: {n_real} | Fake: {n_fake}")
    return n_real, n_fake


def sample(filename_li, labels_d, sample_ratio, balanced=False):
    if balanced:
        examples_real = [filename for filename in filename_li if labels_d[filename] == 0]
        examples_fake = [filename for filename in filename_li if labels_d[filename] == 1]
        n_samples = int(sample_ratio * len(labels_d) * 0.5)

        filenames_sampled = random.sample(examples_real, n_samples) + random.sample(examples_fake, n_samples)
    else:
        n_samples = int(sample_ratio * len(labels_d))
        filenames_sampled = random.sample(filename_li, n_samples)
    return filenames_sampled


def get_pr_score_vecs(weight_mats, G_triple, topic_vecs, pr, idx_twt2idx_usr, news_article_evidence, tweet_df,
                      G_key_score_vec, idx_Gk2bow_id, labels_d, config):
    scores_Gu_d = get_pr_scores_of_nodetype(G_triple, pr, nodetype="usr")
    scores_Gp_d = get_pr_scores_of_nodetype(G_triple, pr, nodetype="twt")

    bow_id2idx_Gk = {v:k for k, v in idx_Gk2bow_id.items()}

    word2topic_weight_mat, topic_dist_mat_sent, topic_dist_mat_post = weight_mats

    topic_dist_mat_post = topic_dist_mat_post.toarray()

    num_posts, num_topics = topic_dist_mat_post.shape
    num_users = config["gat.social"].getint("num_users", 32)
    num_tweets_expected = config["gat.social"].getint("num_tweets_in_each_pair", 6)

    # This is the number of claim-evidence pairs used in KernelGAT
    num_topics_expected = config["KGAT"].getint("evi_num", 5)
    selected_idx_Gp_mat = np.zeros((num_topics, num_tweets_expected), dtype=int)

    news_article_sent_li = news_article_evidence[1]

    assert topic_dist_mat_sent.shape[0] == len(news_article_sent_li)
    assert topic_dist_mat_post.shape[0] == len(tweet_df)

    # Sample topics
    candidate_topics = np.nonzero(np.any(topic_dist_mat_post != 0, axis=0))[0]

    topic_ids_sampled = sample_topics(candidate_topics, config)
    claim_evi_pairs = []

    twt_pos2idx_Gp = dict(zip(np.arange(len(idx_twt2idx_usr)), idx_twt2idx_usr.keys()))

    sent_usage = set([])
    adjacencies = {node_id: nei_dict for node_id, nei_dict in G_triple.adjacency()}

    # Which tweets and users are related to those `topic_ids_sampled`?
    related_idx_Gp_all, related_Gu_all = [], []

    # For each sampled topic
    for i in topic_ids_sampled:

        related_twt_pos = topic_dist_mat_post[:, i].nonzero()[0]

        related_twt_idx = []
        for pos in related_twt_pos:
            if pos in twt_pos2idx_Gp:
                related_twt_idx += [twt_pos2idx_Gp[pos]]

        related_idx_Gp_all += [related_twt_idx]
        related_twt_idx = np.array(related_twt_idx)

        def get_authors_of_post(related_Gp_idx):
            # Here we assume that each tweet is ONLY connected to its author
            related_usr_idx = []
            for idx_Gp in related_Gp_idx:
                post_author_idx = [node_id for node_id, nei_dict in \
                        adjacencies[idx_Gp].items() if G_triple.nodes[node_id]["type"] == "usr"]
                if post_author_idx != []:
                    related_usr_idx += post_author_idx
            related_usr_idx = list(set(related_usr_idx))
            return related_usr_idx

        def get_users_related_to_topic():
            related_bow_ids = word2topic_weight_mat.getrow(1).nonzero()[1]
            related_idx_Gk = [bow_id2idx_Gk[bow_id] for bow_id in related_bow_ids if bow_id in bow_id2idx_Gk]
            related_idx_Gk = np.array(list(set(related_idx_Gk)))
            related_idx_Gu = set([])
            for idx_Gk in related_idx_Gk:
                neighbors = G_triple.neighbors(idx_Gk)
                related_idx_Gu_sublist = [node_id for node_id in \
                        neighbors if G_triple.nodes(data=True)[node_id]["type"]=="usr"]
                related_idx_Gu.update(related_idx_Gu_sublist)
                if len(related_idx_Gu) > num_users:
                    break
            return list(related_idx_Gu)

        # Get users related to topical keywords
        # related_idx_Gu = get_users_related_to_topic()

        # Get only authors of related posts
        related_idx_Gu = get_authors_of_post(related_twt_idx)
        if len(related_idx_Gu) > num_users:
            related_idx_Gu = related_idx_Gu[:num_users]
        elif len(related_idx_Gu) < num_users:
            related_idx_Gu = np.pad(related_idx_Gu, (0, num_users-len(related_idx_Gu)), mode='wrap')

        related_Gu_all += [related_idx_Gu]

        # Get the idx_usr of the author of the tweet

        # We do not pad the feedbacks
        selected_idx_Gp_mat[i, :len(related_twt_idx)] = related_twt_idx[:num_tweets_expected]

        sent_ids_related_to_topic = topic_dist_mat_sent.getcol(i).toarray().nonzero()[0]

        sample_sents(sent_ids_related_to_topic, topic_dist_mat_sent, sent_usage, i, config, claim_evi_pairs,
                 news_article_sent_li, tweet_df, related_twt_idx)


    assert len(related_Gu_all) == len(related_idx_Gp_all)

    related_Gu_all = np.stack(related_Gu_all)

    sim_topics = get_topic_sim_mat(topic_vecs, topic_ids_sampled, config)

    # H_P: num_tweet * num_topics
    twt_weight_mat = np.take_along_axis(topic_dist_mat_post, selected_idx_Gp_mat.T, axis=0)

    # This is for selecting ONLY authors of the posts
    selected_idx_usr_mat = np.vectorize(idx_twt2idx_usr.get)(selected_idx_Gp_mat)

    num_keywords = config["pagerank"].getint("num_words_per_topic", 7)

    # TODO: Now we take the max keyword score for each topic
    # It is better to also consider the weight of each keyword
    # within that topic
    # Ru = np.vectorize(scores_Gu_d.get)(selected_idx_usr_mat)
    # Rk = word2topic_weight_mat.dot(G_key_score_vec).toarray()
    Rp = np.vectorize(scores_Gp_d.get)(selected_idx_Gp_mat)
    Ru = np.vectorize(scores_Gu_d.get)(related_Gu_all)
    Rk = word2topic_weight_mat.todense()[:,:num_keywords]

    twt_weight_mat = twt_weight_mat[:, topic_ids_sampled]
    Rp = Rp[topic_ids_sampled]
    # Ru = Ru[topic_ids_sampled]
    Rk = Rk[topic_ids_sampled]

    def pad_R_scores(R_scores_tmp):
        R_scores = np.zeros((num_topics_expected, R_scores_tmp.shape[1]))
        R_scores[:min(len(R_scores_tmp), num_topics_expected), :] = R_scores_tmp[:num_topics_expected, :]
        return R_scores

    Rp = pad_R_scores(Rp)
    Ru = pad_R_scores(Ru)
    Rk = pad_R_scores(Rk)

    # Rk = np.zeros((num_topics_expected, 1))
    # Rk[min(len(Z_key_tmp), num_topics_expected), :] = Z_key_tmp[:num_topics_expected, :]

    # Z_key_tmp = np.pad(Rk.ravel(), (0, 1), 'constant', constant_values=(0.,))

    assert Rp.shape == (num_topics_expected, num_tweets_expected)
    assert Ru.shape == (num_topics_expected, num_users)
    assert Rk.shape[0] == num_topics_expected

    return (Rp, Ru, Rk), sim_topics, candidate_topics, twt_weight_mat, claim_evi_pairs, (related_Gu_all, related_idx_Gp_all)


def get_pr_scores_of_nodetype(G_triple, pr, nodetype=None):
    '''

    :param G_triple:
    :param nodetype:
    :param attr_score: Whether to return the initial score or final score of PageRank.
            Can be either 'score' or 'final_score'
    :return: dictionary of scores
    '''
    if nodetype is not None:

        node_ids = [node for node in G_triple.nodes if G_triple.nodes[node]['type'] == nodetype]
        score_d = {k: pr[k] for k in node_ids}
        return score_d
    else:
        return pr


def get_sent_topic_weight_dist(model, bow_sents, num_topics, num_sents=None):
    """
    Get the weight distribution of topics within each sentence (tweet)
    :param model:
    :param bow_sents:
    :param num_topics:
    :param num_sents: The number of sentences supposed to be in this example
    :return:
    """
    idx_sent_li, topic_id_li, weight_li = [], [], []
    topic_dist_tup_li = list(model.get_document_topics(bow_sents))
    if num_sents is not None:
        assert len(topic_dist_tup_li) == num_sents
    for idx_sent, row in enumerate(topic_dist_tup_li):
        for topic_id, weight in row:
            idx_sent_li += [idx_sent]  # row
            topic_id_li += [topic_id]  # col
            weight_li += [weight]  # data
    # row: sentence or tweet
    # col: topic_id
    topic_dist_mat = coo_matrix((weight_li, (idx_sent_li, topic_id_li)), shape=(len(topic_dist_tup_li), num_topics))
    return topic_dist_mat


def get_topic_sim_mat(topic_vecs, topic_ids_sampled, config):
    num_topics_expected = config["KGAT"].getint("evi_num", 5)
    topic_vecs = topic_vecs[topic_ids_sampled]
    assert len(topic_vecs) <= num_topics_expected
    sim_mat = np.zeros((num_topics_expected, num_topics_expected))
    sim_topics = cosine_similarity(topic_vecs, topic_vecs)
    sim_mat[:len(topic_vecs), :len(topic_vecs)] = sim_topics
    return sim_mat


def sample_topics(candidate_topics, config):
    """
    Sample topics. DO NOT do padding
    :param candidate_topics:
    :param config:
    :return:
    """
    num_topics_expected = config["KGAT"].getint("evi_num", 5)
    do_padding_topics = config.getboolean("gat.social", "do_padding_topics")

    # If not enough topics, do padding
    if len(candidate_topics) < num_topics_expected:
        if do_padding_topics:
            topic_ids_sampled = np.pad(candidate_topics, pad_width=(0, num_topics_expected - len(candidate_topics)),
                                       mode="wrap")
        else:
            topic_ids_sampled = candidate_topics
    else:
        topic_ids_sampled = np.sort(np.random.choice(candidate_topics, num_topics_expected, replace=False))
    return topic_ids_sampled


def sample_tweets(related_tweet_id, config):
    """
    Sample tweets. If not enough tweets, do padding

    :param related_tweet_id:
    :param config:
    :return:
    """
    num_tweets_expected = config["gat.social"].getint("num_tweets_in_each_pair", 6)
    if len(related_tweet_id) < num_tweets_expected:
        selected_idx_twt = np.pad(related_tweet_id, pad_width=(0, num_tweets_expected - len(related_tweet_id)),
                                  mode="wrap")
    else:
        selected_idx_twt = np.sort(np.random.choice(related_tweet_id, num_tweets_expected, replace=False))
    return selected_idx_twt

def get_adjacent_nodes_of_user(candidate_topics, G_triple, mapping_G_usr, related_idx_usr_all):


    adjacencies = {node_id: nei_dict for node_id, nei_dict in G_triple.adjacency()}

    user_ids = []


    for node_li in related_idx_usr_all:
        for root_node_id in node_li:
            user_ids_wrt_topic = []
            # for node_id, _ in nbrdict.items():
            #     if G_triple.nodes[node_id]["type"] == "usr":
            #         user_ids_wrt_topic += [node_id]
            user_ids += [user_ids_wrt_topic]

    return candidate_user_ids


def sample_sents(sent_ids_related_to_topic, topic_dist_mat_sent, sent_usage, i, config, claim_evi_pairs,
                 news_article_sent_li, tweet_df, related_twt_idx):
    num_sents_in_each_pair = config["gat.social"].getint("num_sents_in_each_pair", 3)
    num_tweets_in_each_pair = config["gat.social"].getint("num_tweets_in_each_pair", 6)

    if sent_ids_related_to_topic.any():
        sent_selection_prob = topic_dist_mat_sent.getcol(i)[sent_ids_related_to_topic].toarray().ravel()
        sent_selection_prob /= sent_selection_prob.sum()

        # Ensure no duplicates in sampling
        size = min(num_sents_in_each_pair, len(sent_ids_related_to_topic))
        sent_ids = np.random.choice(sent_ids_related_to_topic, p=sent_selection_prob, size=size, replace=False)

    else:
        sent_ids = np.random.choice(np.arange(topic_dist_mat_sent.shape[0]), size=num_sents_in_each_pair)

    # If no related claims, sample random news sentence as claim
    # if not sent_ids.any():
    #     sent_ids = np.random.choice()

    sent = [news_article_sent_li[sent_id] for sent_id in sent_ids]
    sent = " ".join(sent)
    tweets = tweet_df.iloc[related_twt_idx].text.to_list()
    # Padding
    if len(tweets) < num_tweets_in_each_pair and not tweet_df.text.to_list() == []:
        tweets += random.choices(tweet_df.text.to_list(), k=num_tweets_in_each_pair-len(tweets))
    tweets = tweets[:num_tweets_in_each_pair]
    claim_evi_pairs += [[sent, tweets]]
    sent_usage.update(sent_ids.tolist())

def select_claim_evi_pairs(tweet_df, news_articles, config):
    # num_users = config["gat.social"].getint("num_users", 32)
    num_topics_expected = config["KGAT"].getint("evi_num", 5)
    num_sents_in_each_pair = config["gat.social"].getint("num_sents_in_each_pair", 3)
    num_tweets = config["gat.social"].getint("num_tweets_in_each_pair", 6)
    num_words_per_topic = config["pagerank"].getint("num_words_per_topic", 7)
    num_users = config["gat.social"].getint("num_users", 32)

    claim_evi_pairs = []
    Rs = []

    for i in range(num_topics_expected):
        #if news_articles != []:
        sents = random.choices(news_articles, k=num_sents_in_each_pair)
        sents = " ".join(sents)
        tweets = random.choices(tweet_df.text.to_list(), k=num_tweets)
        Rs += [get_Rs_paddings(num_topics_expected, num_tweets, num_users, num_words_per_topic)]
        claim_evi_pairs += [sents, tweets]


    return Rs, claim_evi_pairs


def get_Rs_paddings(num_topics_expected, num_tweets, num_users, num_words_per_topic, mode="random"):
    """
    Zero-pad the R scores
    :param num_topics_expected:
    :param num_tweets:
    :param num_users:
    :param num_words_per_topic:
    :return: MR scores of post, users, and keywords
    """

    if mode == "random":
        mu = 1e-3
        std = 1e-3
        R_p = torch.normal(mean=mu, std=std, size=(num_topics_expected, num_tweets))
        R_u = torch.normal(mean=mu, std=std, size=(num_topics_expected, num_users))
        R_k = torch.normal(mean=mu, std=std, size=(num_topics_expected, num_words_per_topic))

        R_p = torch.clamp(R_p, min=5e-4)
        R_u = torch.clamp(R_u, min=5e-4)
        R_k = torch.clamp(R_k, min=5e-4)
        return [R_p, R_u, R_k]
    else:

        R_p = torch.zeros(size=(num_topics_expected, num_tweets))
        R_u = torch.zeros(size=(num_topics_expected, num_users))
        R_k = torch.zeros(size=(num_topics_expected, num_words_per_topic))
    return [R_p, R_u, R_k]

