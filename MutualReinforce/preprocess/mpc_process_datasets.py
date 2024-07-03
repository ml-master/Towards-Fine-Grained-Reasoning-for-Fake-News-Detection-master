import json
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool, Manager

import pandas as pd


from mpc_process_graph import process_retweet, filter_empty_dict_entry, process_comment
from process_text import preprocess_tweet
from config import *


import sys
sys.path.append("..")
from utils.utils import only_directories



DEBUG = False
#done

def get_news_article_title(news_article_json):
    if "title" in news_article_json:
        return news_article_json["title"]
    elif "meta_data" in news_article_json:
        meta_data = news_article_json["meta_data"]
        if "og" in meta_data and "title" in meta_data["og"]:
            return meta_data["og"]["title"]
        elif "twitter" in meta_data and "title" in meta_data["twitter"]:
            return meta_data["twitter"]["title"]


def process_example(filename, dataset_name, global_news_article_d):
    print(filename)
    dataset_full_path = FAKE_NEWS_DATA + "\\" + dataset_name
    example_name = dataset_full_path + "\\" + filename

    user_features_d = {}

    ########################
    # News Article
    ########################

    with open(example_name + '\\news_article.json', 'r', encoding='utf-8') as f:
        news_json = json.loads(f.read())
        f.close()

    has_news_article = TRUE
    news_text = EMPTY_STR
    news_title = EMPTY_STR
    news_timestamp = INVALID_TIMESTAMP

    # Abandon upon empty news article
    if news_json == {}:
        has_news_article = FALSE

    # It's possible that only the format is unusual
    elif news_json["text"] == EMPTY_STR:

        if "meta_data" in news_json:
            meta_data = news_json["meta_data"]
            if "description" in meta_data:
                news_text = meta_data["description"]
            elif "og" in meta_data:
                og = meta_data["og"]
                if "description" in og:
                    news_text = og["description"]
            elif "twitter" in meta_data:
                twitter = meta_data["twitter"]
                if "description" in twitter:
                    news_text = twitter["description"]
        else:
            has_news_article = FALSE
    else:
        news_text = news_json["text"]
        news_timestamp = news_json["publish_date"]

    news_text = preprocess_tweet(news_text)
    if news_text == "":
        has_news_article == FALSE

    if has_news_article:
        news_title = preprocess_tweet(get_news_article_title(news_json))
    else:
        print(f"\t{filename} news is empty")

    news_article_d = {
        "id": filename,
        "label": label_d[dataset_name],
        "title": news_title,
        "text": preprocess_tweet(news_text),
        "timestamp": news_timestamp,
        "has_news_article": has_news_article,
        "has_tweets": FALSE,
        "has_retweets": FALSE,
        "has_replies": FALSE
    }

    # Abandon this example if news article is empty
    if not has_news_article:
        global_news_article_d[filename] = news_article_d
        return

    ########################
    # Tweets
    ########################

    with open(example_name + '\\tweets.json', 'r', encoding='utf-8') as f_tweets:
        tweet_json = json.loads(f_tweets.read())
        f_tweets.close()
    tweet_json = tweet_json["tweets"]

    # Abandon upon empty tweets
    if tweet_json == []:

        # Append example to list
        print(f"\t{filename} news exists, tweets is empty")
        news_article_d["has_tweets"] = FALSE
        global_news_article_d[filename] = news_article_d
        return
    else:
        for tweet in tweet_json:
            tweet["text"] = preprocess_tweet(tweet["text"])
            user_features_d[tweet["user_id"]] = {
                "id": tweet["user_id"],
                "is_tweeter": TRUE
            }
        news_article_d["has_tweets"] = TRUE
    tweet_retweet_comment_df = pd.DataFrame.from_dict(tweet_json).drop(['user_name'], axis=1)
    tweet_retweet_comment_df["type"] = TYPE_TEXT["tweet"]

    # Root node id is set to 0
    tweet_retweet_comment_df["root_tweet_id"] = 0
    tweet_retweet_comment_df["root_user_id"] = 0

    ########################
    # Retweets
    ########################

    with open(example_name + '\\retweets.json', 'r', encoding='utf-8') as f:
        retweet_json = json.loads(f.read())
        f.close()

    retweet_json, is_empty_json = filter_empty_dict_entry(retweet_json, "retweet")

    if is_empty_json:
        news_article_d["has_retweets"] = FALSE
        retweet_df = None
    else:
        news_article_d["has_retweets"] = TRUE

        retweet_list = []

        process_retweet(retweet_json, filename, tweet_retweet_comment_df, retweet_list, user_features_d)

        retweet_df = pd.DataFrame.from_dict(retweet_list)

    ########################
    # Comments
    ########################

    with open(example_name + '\\replies.json', 'r', encoding='utf-8') as f:
        comment_json = json.loads(f.read())
        f.close()

    comment_json, is_empty_json = filter_empty_dict_entry(comment_json, "comment")

    if is_empty_json:
        news_article_d["has_replies"] = FALSE
        comment_df = None
    else:
        news_article_d["has_replies"] = TRUE

        comment_list = []

        process_comment(comment_json, filename, tweet_retweet_comment_df, comment_list, user_features_d)

        comment_df = pd.DataFrame.from_dict(comment_list)

    if retweet_df is not None:
        tweet_retweet_comment_df = tweet_retweet_comment_df.append(retweet_df)

    if comment_df is not None:
        tweet_retweet_comment_df = tweet_retweet_comment_df.append(comment_df)

    ########################
    # Results Summary
    ########################

    global_news_article_d[filename] = news_article_d

    tweet_retweet_comment_df.to_csv(example_name + '\\tweets_retweets_comments.tsv', sep='\t', index=False)

    ########################
    # Users
    ########################

    users_df = pd.DataFrame.from_dict(user_features_d).transpose()

    for key in USER_IDENTITY_KEYS:
        if key in users_df:
            users_df[key].fillna(value=FALSE, inplace=True)
        else:
            users_df[key] = FALSE
    users_df.to_csv(example_name + '\\old_user.tsv', sep='\t', index=False)


def main():
    politifact_fake = only_directories(POLITIFACT_FAKE)
    politifact_real = only_directories(POLITIFACT_REAL)
    gossipcop_fake = only_directories(GOSSIPCOP_FAKE)
    gossipcop_real = only_directories(GOSSIPCOP_REAL)

    data_list = {
        #POLITIFACT_REAL_NAME: politifact_real,
        #POLITIFACT_FAKE_NAME: politifact_fake,

        GOSSIPCOP_FAKE_NAME: gossipcop_fake,
        GOSSIPCOP_REAL_NAME: gossipcop_real
    }

    manager = Manager()

    n_cpu = mp.cpu_count()

    print(f"Using {str(n_cpu)} processors")

    # dataset_name is "politifact" or "gossipcop"
    for dataset_name, dataset_list in DATASET_NAMES.items():
        print("#" * 30 + f"\n# Processing {dataset_name}\n" + "#" * 30 + "\n")

        global_news_article_d = manager.dict()

        if DEBUG:
            job_list = [["politifact15501", POLITIFACT_FAKE_NAME],
                        ['politifact11773', POLITIFACT_FAKE_NAME]]

        else:
            job_list = []
            for dataset in dataset_list:
                for filename in data_list[dataset]:
                    job_list += [[filename, dataset]]

        pool = Pool(processes=n_cpu)



        partial_process_example = partial(process_example, global_news_article_d=global_news_article_d)

        pool.starmap(partial_process_example, job_list)

        f = open(FAKE_NEWS_DATA + f"\\{dataset_name}_news_articles.txt", "w", encoding='utf-8')

        for filename, article_json in global_news_article_d.items():
            f.write(filename + "\t" + article_json['text'] + '\n')
            article_json['text'] = EMPTY_STR
        f.close()

        global_news_article_df = pd.DataFrame.from_dict(global_news_article_d.values()).drop(['text'], axis=1)

        for key in global_news_article_df:
            if key in global_news_article_df:
                global_news_article_df[key].fillna(value=FALSE, inplace=True)
            else:
                global_news_article_df[key] = FALSE

        global_news_article_df.to_csv(FAKE_NEWS_DATA + f"\\{dataset_name}_news_articles.tsv", sep="\t", index=False)

        f.close()

        if DEBUG:
            break


if __name__ == "__main__":
    main()
