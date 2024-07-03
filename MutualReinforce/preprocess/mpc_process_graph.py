import os
import sys
from datetime import datetime
from config import *
import numpy as np

from download_user_features import extract_user_features
from process_text import preprocess_tweet

# This is to handle cases in which parent tweet id is not in the original twee.json
# We perform a prefix matching
def get_possible_parent_tweet_id(retweet_text, filename, tweet_df):
    SEARCH_PREFIX_LEN = 10
    try:
        root_tweet_id = tweet_df[tweet_df['text'].str.startswith(retweet_text.split('RT ')[1][:SEARCH_PREFIX_LEN])].tweet_id
        return root_tweet_id.iloc[0]
    except Exception as e:
        print(f"\t{filename} Tweet NOT FOUND in original tweet.json")
        print(f"\t{retweet_text[:30]}")
        print(e)

def filter_empty_dict_entry(d, filename):
    is_empty_json = False
    new_d = {}
    for k in d:
        if d[k] != []:
            new_d[int(k)] = d[k]
    if new_d == {}:
        print(f"\t{filename} json empty")
        is_empty_json = True

    return new_d, is_empty_json


def process_retweet(retweet_json, filename, tweet_df, retweet_list, user_features_d):
    '''
    Process a single example
    :param retweet_json: dictionary read from retweet.json file
    :param filename: Name of the samle
    :param tweet_df:
    :param retweet_list:
    :param user_features_d:
    :return:
    '''

    parent_tweet_id_matching_cache = {}

    def nested_lookup_user(retweet_json_nested, potential_tweet_id):
        nonlocal retweet_list, parent_tweet_id_matching_cache

        assert "user" in retweet_json_nested
        assert "id_str" in retweet_json_nested
        assert "retweeted_status" in retweet_json_nested

        curr_user = retweet_json_nested["user"]

        curr_tweet_id = int(retweet_json_nested["id_str"])
        curr_user_id = int(curr_user["id_str"])
        curr_retweet_text = preprocess_tweet(retweet_json_nested["text"])
        created_at = retweet_json_nested["created_at"]

        user_features_d[curr_user_id] = extract_user_features(curr_user)
        user_features_d[curr_user_id]['is_retweeter'] = TRUE

        retweet_status = retweet_json_nested["retweeted_status"]

        if potential_tweet_id in tweet_df['tweet_id']:
            root_tweet_id = potential_tweet_id
            root_user_id = tweet_df[tweet_df['tweet_id'] == root_tweet_id]["user_id"]
        else:
            root_tweet_id = int(retweet_status["id_str"])
            root_user_id  = int(retweet_status["user"]["id_str"])


        if not root_tweet_id or root_tweet_id not in tweet_df["tweet_id"].values:

            # If we have matched this prefix before
            if root_tweet_id in parent_tweet_id_matching_cache:
                print(f"\t{root_tweet_id} Using CACHED prefix")
                root_tweet_id_matched = parent_tweet_id_matching_cache[root_tweet_id]

            # Else this is the first time we encounter this prefix
            else:
                print(f"\t{root_tweet_id} prefix matching")
                root_tweet_id_matched = get_possible_parent_tweet_id(curr_retweet_text, filename, tweet_df)
            if root_tweet_id_matched:
                parent_tweet_id_matching_cache[root_tweet_id] = root_tweet_id_matched

        assert curr_tweet_id != root_tweet_id
        # assert curr_user_id != root_user_id

        retweet_list += [{
            "tweet_id": curr_tweet_id,
            "root_tweet_id": root_tweet_id,
            "user_id": curr_user_id,
            "root_user_id": root_user_id,
            "text": curr_retweet_text,
            "created_at": created_at,
            "type": TYPE_TEXT["retweet"]
        }]

    # Not sure if each key in retweet.json is the potential tweet_id
    for potential_tweet_id in retweet_json:
        for retweet in retweet_json[potential_tweet_id]:
            nested_lookup_user(retweet, int(potential_tweet_id))

def process_comment(comment_json, filename, tweet_retweet_comment_df, comment_list, user_features_d):


    def nested_lookup_user(comment_json_nested, potential_tweet_id=None, root_user_id=None):
        '''

        :param comment_json_nested:
        :param potential_tweet_id: Only first-level replies can have potential_tweet_id
        :param root_user_id:
        :return:
        '''

        nonlocal comment_list
        curr_comment_id = int(comment_json_nested["id"])

        # Retrieve current user_id
        if "user_id" not in comment_json_nested:
            curr_user = comment_json_nested["user"]
            if type(curr_user) == int:
                curr_user_id = curr_user
            else:
                curr_user_id = int(curr_user["id_str"])
                user_features_d[curr_user_id] = extract_user_features(curr_user)
        else:
            curr_user_id = int(comment_json_nested["user_id"])
        user_features_d[curr_user_id] = {
            "id": curr_user_id,
            "is_replier": TRUE
        }


        curr_comment_text = preprocess_tweet(comment_json_nested["text"])

        created_at = comment_json_nested['created_at']

        root_tweet_id = comment_json_nested["in_reply_to_status_id"]

        if not root_tweet_id:
            root_tweet_id = comment_json_nested["id"]

        # Process first-level comment directly targeted at a tweet
        if not root_user_id:

            entry = tweet_retweet_comment_df[tweet_retweet_comment_df["tweet_id"] == int(root_tweet_id)]
            if not entry.empty and 'user_id' in entry:
                root_user_id = entry.user_id.item()
            else:
                entry = tweet_retweet_comment_df[tweet_retweet_comment_df["tweet_id"] == int(potential_tweet_id)]
                if not entry.empty and 'user_id' in entry:
                    root_user_id = entry.user_id.item()
                else:
                    print(f"\tIn {filename} root user {root_user_id} empty")
            # tweet_user_id = int(tweet_df[tweet_df["tweet_id"] == int(root_tweet_id)].user_id.item())

        # Abandon node
        if root_tweet_id is None:
            print(f"\tEmpty Tweet ID {root_tweet_id}")
            return
        if root_user_id is None:
            print(f"\tEmpty User  ID {root_user_id}")
            return

        comment_list += [{
            "tweet_id": curr_comment_id,
            "root_tweet_id": root_tweet_id,
            "user_id": curr_user_id,
            "root_user_id": root_user_id,
            "text": curr_comment_text,
            "created_at": created_at,
            "type": TYPE_TEXT["reply"]
        }]

        # If there is nested comments
        if "engagement" in comment_json_nested:

            # tweet_replies
            comment_json_nested_new = comment_json_nested["engagement"]["tweet_replies"]
            for comment in comment_json_nested_new:
                if comment is not None:
                    nested_lookup_user(comment, root_user_id=curr_user_id)

            # tweet_retweets
            comment_json_nested_new = comment_json_nested["engagement"]["tweet_retweets"]
            for comment in comment_json_nested_new:
                if comment is not None:
                    nested_lookup_user(comment, root_user_id=curr_user_id)

    for tweet_id in comment_json:
        for comment in comment_json[tweet_id]:

            # Key is the index of the reply list, which may be the tweet id
            # Empty comment
            if comment is not None:
                nested_lookup_user(comment, potential_tweet_id=tweet_id)


# Create 10-feature dataframe
def hand_feature(user_dict):

    feature = np.zeros([len(user_dict), 11], dtype=np.float64)
    id_counter = 0
    est_date = datetime.fromisoformat('2021-06-03')
    for profile in user_dict.values():

        try:
            vector = [profile['id']]

            # 1) Verified?, 2) Enable geo-spatial positioning, 3) Followers count, 4) Friends count
            vector += [int(profile['verified']), int(profile['geo_enabled']), profile['followers_count'], profile['friends_count']]
            # 5) Status count, 6) Favorite count, 7) Number of lists
            vector += [profile['statuses_count'], profile['favourites_count'], profile['listed_count']]

            # 8) Created time (No. of months since Twitter established)
            user_date = datetime.strptime(profile['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            month_diff = (user_date.year - est_date.year) * 12 + user_date.month - est_date.month
            vector += [month_diff]

            # 9) Number of words in the description, 10) Number of words in the screen name
            vector += [len(profile['name'].split()), len(profile['description'].split())]

            feature[id_counter, :] = np.reshape(vector, (1, 11))
            id_counter += 1
            # print(id_counter)
        except Exception as e:

            print("Error: field missing")
            print(e)

    return feature
