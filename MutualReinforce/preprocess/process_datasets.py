import json
import os
import pickle

import pandas as pd

from config import *
from process_graph import process_retweet_users, filter_empty_dict_entry, process_comments
from process_text import preprocess_tweet

def only_directories(path):
    return [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

def get_news_article_title(news_article_json):
    if "title" in news_article_json:
        return news_article_json["title"]
    elif "meta_data" in news_article_json:
        meta_data = news_article_json["meta_data"]
        if "og" in meta_data and "title" in meta_data["og"]:
            return meta_data["og"]["title"]
        elif "twitter" in meta_data and "title" in meta_data["twitter"]:
            return meta_data["twitter"]["title"]


def main():

    #politifact_fake = only_directories(POLITIFACT_FAKE)
    #politifact_real = only_directories(POLITIFACT_REAL)
    gossipcop_fake = only_directories(GOSSIPCOP_FAKE)
    gossipcop_real = only_directories(GOSSIPCOP_REAL)

    data_list = {
        #"politifact_real": politifact_real,
        #"politifact_fake": politifact_fake,
        "gossipcop_fake": gossipcop_fake,
        "gossipcop_real": gossipcop_real
    }



    for dataset_name, dataset in data_list.items():
        print("#"*30 + f"\n# Processing {dataset_name}\n" + "#"*30 + "\n")

        # Stores all tweeters' user_ids so that we can later crawl the data
        global_user_ids = {}

        # Stores all retweeters' user features, just in case they cannot be crawled through twitter API
        global_user_features_d = {}
        global_comment_d = {}
        global_news_article_d = []


        n_empty = 0

        list_only_news_article, list_normal = [], []

        dataset_full_filename = FAKE_NEWS_DATA + "/" + dataset_name

        list_only_news_article_filename = dataset_full_filename + "_news_only.txt"
        list_filename = FAKE_NEWS_DATA + dataset_name + ".txt"


        for filename in dataset:
            print(filename)
            example_name = dataset_full_filename + "/" + filename

            if os.path.exists(example_name + "/tweets.tsv"):
                os.remove(example_name + "/tweets.tsv")


            user_features_d = {}

            ########################
            # News Article
            ########################
            news_json ={}
            # with open(example_name + '/news_article.json', 'r', encoding='utf-8') as f:
            #     news_json = json.loads(f.read())
            #     f.close()

            # Abandon upon empty news article
            if news_json == {}:
                print(f"\t{filename} news is empty")
                n_empty += 1
                continue



            if news_json["text"] == "":
                if "meta_data" in news_json:
                    meta_data = news_json["meta_data"]
                    if "description" in meta_data:
                        news_text = meta_data["description"]
                else:
                    continue

            else:
                news_text = news_json["text"]
            news_timestamp = news_json["publish_date"]


            ########################
            # Tweets
            ########################

            with open(example_name + '/tweets.json', 'r', encoding='utf-8') as f_tweets:
                tweet_json = json.loads(f_tweets.read())
                f_tweets.close()
            tweet_json = tweet_json["tweets"]

            # Abandon upon empty tweets
            if tweet_json == []:
                # Append example to list
                print(f"\t{filename} news exists, tweets is empty")
                list_only_news_article += [filename]
                continue
            else:
                for tweet in tweet_json:
                    tweet["text"] = preprocess_tweet(tweet["text"])
                    user_features_d[tweet["user_id"]] = {
                        "id": tweet["user_id"],
                        "is_tweeter": True
                    }

            # Global news article collector Shared by all examples
            # Later we sort it by timestamp
            # Empty examples are discarded
            if news_text != "":
                global_news_article_d += [{
                    "id":filename,
                    "timestamp": news_timestamp,
                    "title": preprocess_tweet(get_news_article_title(news_json)),
                    "text": preprocess_tweet(news_text),
                    "label": label_d[dataset_name]
                }]

            tweet_retweet_comment_df = pd.DataFrame.from_dict(tweet_json).drop(['user_name'], axis=1)

            tweet_retweet_comment_df["type"] = "tweet"

            # Root node id is set to 0
            tweet_retweet_comment_df["root_tweet_id"] = 0
            tweet_retweet_comment_df["root_user_id"] = 0


            user_ids = tweet_retweet_comment_df["user_id"].to_list()



            # Add node of tweets and dummy source news article

            root_node_id = filename
            edges_user  = [[root_node_id, user_id] for user_id in tweet_retweet_comment_df.user_id]
            edges_tweet = [[root_node_id, tweet_id] for tweet_id in tweet_retweet_comment_df.tweet_id]
            nodes_user = set(tweet_retweet_comment_df.user_id); nodes_user.add(filename)
            nodes_tweet = set(tweet_retweet_comment_df.tweet_id); nodes_tweet.add(filename)


            ########################
            # Retweets
            ########################

            with open(example_name + '/retweets.json', 'r', encoding='utf-8') as f:
                retweet_json = json.loads(f.read())
                f.close()

            retweet_json, is_empty_json = filter_empty_dict_entry(retweet_json, "retweet")

            retweet_list = []

            process_retweet_users(retweet_json, filename, nodes_user, nodes_tweet, edges_user, edges_tweet, tweet_retweet_comment_df, retweet_list, user_features_d)

            ########################
            # Comments
            ########################

            with open(example_name + '/replies.json', 'r', encoding='utf-8') as f:
                comment_json = json.loads(f.read())
                f.close()

            comment_json, is_empty_json = filter_empty_dict_entry(comment_json, "comment")

            comment_list = []

            process_comments(comment_json, filename, nodes_user, nodes_tweet, edges_user, edges_tweet, tweet_retweet_comment_df, comment_list, user_features_d)


            retweet_df = pd.DataFrame.from_dict(retweet_list)
            comment_df = pd.DataFrame.from_dict(comment_list)


            tweet_retweet_comment_df = tweet_retweet_comment_df.append(retweet_df).append(comment_df)

            global_comment_d[filename] = comment_list

            user_ids += nodes_user
            user_ids.remove(filename)

            # remove duplicates so that it is easier to crawl features
            global_user_ids[filename] = list(set(user_ids))

            tweet_retweet_comment_df.to_csv(f'{example_name}/tweets_retweets_comments.tsv', sep = '\t', index=False)

            global_user_features_d[filename] = user_features_d

            # [u['is_replier'] or u['is_retweeter'] or u['is_retweeter'] for _, u in user_features_d.items()]

            # Append example to list
            list_normal += [filename]

            users_df = pd.DataFrame.from_dict(user_features_d).transpose()

            for key in USER_IDENTITY_KEYS:
                if key in users_df:
                    users_df[key].fillna(value=False, inplace=True)
                else:
                    users_df[key] = False
            # users_df.is_tweeter.fillna(value=False, inplace=True)
            # users_df.is_retweeter.fillna(value=False, inplace=True)
            # users_df.is_replier.fillna(value=False, inplace=True)
            users_df.to_csv(example_name+'/old_user.tsv', sep = '\t', index=False)

        f = open(dataset_full_filename + "/all_user_ids_and_features.pkl", "wb")
        pickle.dump((global_user_ids, global_user_features_d),f)
        f.close()

        f = open(f"{dataset_full_filename}/{dataset_name}_news_articles.pickle", "wb")
        pickle.dump(global_news_article_d,f)
        f.close()

        f = open(f"{dataset_full_filename}/only_news_article.txt", "w")
        for filename in list_only_news_article:
            f.write(filename + '\n')
        f.close()

        f = open(dataset_full_filename + "/normal.txt", "w")
        for filename in list_normal:
            f.write(filename + '\n')
        f.close()

        print(f"[{dataset_name}] | Empty News: {n_empty} ({float(n_empty) / (len(dataset) * 100 +1): .2f}%) | Only News: {len(list_only_news_article)} ({float(len(list_only_news_article)) / (len(dataset) * 100 + 1) : .2f}%)")
        

if __name__ == "__main__":
    main()
