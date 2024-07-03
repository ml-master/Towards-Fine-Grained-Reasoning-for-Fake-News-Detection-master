from datetime import datetime

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


def process_retweet_users(retweet_json, filename, nodes_user, nodes_tweet, edges_user, edges_tweet, tweet_df, retweet_list, user_features_d, print_results=True):

    def nested_lookup_user(retweet_json_nested, potential_tweet_id):
        nonlocal retweet_list

        assert "user" in retweet_json_nested
        assert "id_str" in retweet_json_nested
        assert "retweeted_status" in retweet_json_nested

        curr_user = retweet_json_nested["user"]
        curr_tweet_id = int(retweet_json_nested["id_str"])
        curr_user_id = int(curr_user["id_str"])
        curr_retweet_text = preprocess_tweet(retweet_json_nested["text"])
        created_at = retweet_json_nested["created_at"]

        user_features_d[curr_user_id] = extract_user_features(curr_user)
        user_features_d[curr_user_id]['is_retweeter'] = True

        retweet_status = retweet_json_nested["retweeted_status"]

        if potential_tweet_id in tweet_df['tweet_id']:
            root_tweet_id = potential_tweet_id
            root_user_id = tweet_df[tweet_df['tweet_id'] == root_tweet_id]["user_id"]
        else:
            root_tweet_id = int(retweet_status["id_str"])
            root_user_id  = int(retweet_status["user"]["id_str"])


        if not root_tweet_id or root_tweet_id not in tweet_df["tweet_id"].values:
            print(f"\t{root_tweet_id} Using prefix matching")
            root_tweet_id = get_possible_parent_tweet_id(curr_retweet_text, filename, tweet_df)

        assert curr_tweet_id != root_tweet_id
        # assert curr_user_id != root_user_id

        retweet_list += [{
            "tweet_id": curr_tweet_id,
            "root_tweet_id": root_tweet_id,
            "user_id": curr_user_id,
            "root_user_id": root_user_id,
            "text": curr_retweet_text,
            "created_at": created_at,
            "type": "retweet"
        }]

        # try:
        #     tweet_user_id = int(tweet_df[tweet_df["tweet_id"] == int(root_tweet_id)].user_id.item())
        # except Exception as e:
        #     print(f"\tError: {filename} abandon retweet node")
        #     return
        # if not root_user_id:
        #     root_user_id = tweet_user_id
        # else:
        #     if tweet_user_id != root_user_id:
        #         print()
        #     assert tweet_user_id == root_user_id, "Error: tweet_user_id != root_tweet_id"

        # Add edge from tweet -> retweet
        edges_user.append([root_user_id, curr_user_id])
        edges_tweet.append([root_tweet_id, curr_tweet_id])

        nodes_user.add(curr_user_id)
        nodes_tweet.add(curr_tweet_id)



    # Not sure if each key in retweet.json is the potential tweet_id
    for potential_tweet_id in retweet_json:
        for retweet in retweet_json[potential_tweet_id]:
            nested_lookup_user(retweet, int(potential_tweet_id))

    if print_results:
        print(f"\t[Retweets] User:  Nodes: {len(nodes_user):5} | Edges: {len(edges_user):5} | Tweet: Nodes: {len(nodes_tweet):5} | Edges: {len(edges_tweet):5}")

    return

def process_comments(comment_json, filename, nodes_user, nodes_tweet, edges_user, edges_tweet, tweet_retweet_comment_df, comment_list, user_features_d, print_results=True):

    def nested_lookup_user(comment_json_nested, key=None, root_user_id=None, is_first_level_comment=False):
        nonlocal comment_list
        try:

            # curr_user = comment_json_nested["user_id"]
            curr_comment_id = int(comment_json_nested["id"])
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
                "is_replier": True
            }


            curr_comment_text = preprocess_tweet(comment_json_nested["text"])

            created_at = comment_json_nested['created_at']

            root_tweet_id = comment_json_nested["in_reply_to_status_id"]
            if is_first_level_comment and not root_user_id:
                try:
                    root_user_id = tweet_retweet_comment_df[tweet_retweet_comment_df["tweet_id"] == int(root_tweet_id)].user_id
                    if root_user_id.empty:
                        root_user_id = tweet_retweet_comment_df[tweet_retweet_comment_df["tweet_id"] == int(tweet_id)].user_id
                    root_user_id = root_user_id.item()
                    # tweet_user_id = int(tweet_df[tweet_df["tweet_id"] == int(root_tweet_id)].user_id.item())
                except Exception as e:
                    print(f"\tError: {filename} abandon retweet node")
                    return

            comment_list += [{
                "tweet_id": curr_comment_id,
                "root_tweet_id": root_tweet_id,
                "user_id": curr_user_id,
                "root_user_id": root_user_id,
                "text": curr_comment_text,
                "created_at": created_at,
                "type": "comment"
            }]


            edges_user.append([root_user_id, curr_user_id])
            edges_tweet.append([root_tweet_id, curr_comment_id])

            nodes_user.add(curr_user_id)
            nodes_tweet.add(curr_comment_id)

            # If there is nested comments
            if "engagement" in comment_json_nested:

                # tweet_replies
                comment_json_nested_new = comment_json_nested["engagement"]["tweet_replies"]
                for comment in comment_json_nested_new:
                    nested_lookup_user(comment, curr_user_id)

                # tweet_retweets
                comment_json_nested_new = comment_json_nested["engagement"]["tweet_retweets"]
                for comment in comment_json_nested_new:
                    nested_lookup_user(comment, curr_user_id)
        except Exception as e:
            print(e)

    for tweet_id in comment_json:
        for comment in comment_json[tweet_id]:
            # tweet_user_id = tweet_df[tweet_df["tweet_id"] == int(tweet_id)].user_id.item()
            # Key is the index of the reply list, which may be the tweet id
            nested_lookup_user(comment, key=tweet_id, is_first_level_comment=True)

    if print_results:
        print(f"\t[Comments] User:  Nodes: {len(nodes_user):5} | Edges: {len(edges_user):5} | Tweet: Nodes: {len(nodes_tweet):5} | Edges: {len(edges_tweet):5}")

    return



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
