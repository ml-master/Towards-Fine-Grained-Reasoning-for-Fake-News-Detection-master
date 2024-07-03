import json
import re
from collections import Counter

import pandas as pd

def preprocess_tweet(text):
    # Substitute URL
    tweet = re.sub(r'https?:\/\/[\w_\.\/\-\?&=]*[\s\r\n]?', '', text, flags=re.MULTILINE)
    # Substitute '@'
    tweet = re.sub(r'@[\w]*[\s\:\,\.]|@[\w]*$', '', tweet, flags=re.MULTILINE)

    # Substitute '#' hashtags
    tweet = re.sub(r'#[\w_\.]*', '', tweet, flags=re.MULTILINE)

    # Remove extra whitespaces
    tweet = re.sub(r'[\s]+', ' ', tweet)

    # Remove trailing whitespaces
    tweet = re.sub(r'[\s]$', '', tweet)

    # Replace extra UTF characters
    tweet = re.sub(u"\u2013", "-", tweet)
    tweet = re.sub(u"\u2019", "\'", tweet)


    tweet = tweet.replace('’', '\'', ).replace('“', '\"', ).replace('”', '\"', ).replace('…', '...')

    # Add trailing whitespace to '.'
    # text = re.sub('\.[\s]*', '. ', text)

    return tweet.strip()


def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words


def main():
    dict_politi_real_example = dict()

    path_tweet = "{}/tweets.json".format(path_example)

    node_id = 0
    # Tweets
    with open(path_tweet, 'r', encoding='utf-8') as f:
        lines = f.read()
        json_tweets = json.loads(lines)
        if "tweets" in json_tweets:
            for tweet in json_tweets["tweets"]:
                text = preprocess_tweet(tweet["text"])
                tweet_id = tweet["tweet_id"]
                parent_tweet_id = example_id
                # tweet_user_id = tweet["user_id"]
                dict_politi_real_example[tweet_id] = {
                    'text': text, 'tweet_id': tweet_id, "node_id": node_id,

                    'parent_tweet_id': parent_tweet_id
                }
                node_id += 1

    # Retweets
    path_retweet = "{}/retweets.json".format(path_example)
    with open(path_retweet, 'r', encoding='utf-8') as f:
        lines = f.read()
        json_retweet = json.loads(lines)
        for id in json_retweet:
            if json_retweet[id] != []:
                for retweet in json_retweet[id]:
                    text = preprocess_tweet(retweet['text'])
                    tweet_id = retweet["id"]
                    parent_tweet_id = retweet['in_reply_to_status_id']
                    dict_politi_real_example[tweet_id] = {
                        'text': text, 'tweet_id': tweet_id, "node_id": node_id,

                        'parent_tweet_id': id
                    }
                    node_id += 1

    # Replies
    path_reply = "{}/replies.json".format(path_example)

    with open(path_reply, 'r', encoding='utf-8') as f:
        lines = f.read()
        json_reply = json.loads(lines)

        def user_engagement(d, parent_tweet_id):
            if d == []:
                return
            dict_politi_real_example[tweet_id] = {

                'text': preprocess_tweet(json['text']),
                'tweet_id': d['tweet_id'],
                "node_id": node_id,
                'parent_tweet_id': parent_tweet_id
            }
            node_id += 1
            if "engagement" in d:
                user_engagement(d["engagement"], tweet_id)

        for id in json_reply:
            if json_reply[id] != []:

                # Get number of replies corresponding to some retweet
                n_replies = len(json_reply[id])

                # The retweet does NOT exist in retweet.json
                if id not in dict_politi_real_example:
                    dict_politi_real_example[tweet_id] = {
                        'text': text, 'tweet_id': id, "node_id": node_id, 'parent_tweet_id': example_id
                    }
                    node_id += 1

                for reply in json_reply[id]:
                    user_engagement(reply)
                    # text = preprocess_tweet(reply['text'])
                    # tweet_id = reply["id"]
                    # parent_tweet_id = reply['in_reply_to_status_id']
                    # dict_politi_real_example[tweet_id] = {'text': text, 'tweet_id': tweet_id, "node_id": node_id,
                    #
                    #                                       'parent_tweet_id': parent_tweet_id}
                    # node_id += 1

    df = pd.DataFrame(dict_politi_real_example).T

    # Create Bag-of-word vector
    df['vec'] = None

    # for sentence in df.text:
    vocab = tokenize(df.text)

    dict_id2word = {id: word for (id, word) in enumerate(vocab)}
    dict_word2id = {word: id for (id, word) in enumerate(vocab)}

    for id, sentence in df.text.items():
        words = word_extraction(sentence)
        counter = Counter(words)

        # Format: (word_id:frequency)
        word_frequency = {dict_word2id[word]: counter[word] for word in counter}
        str(word_frequency, )
        word_vec = " ".join(f"{key}:{value}" for key, value in word_frequency.items())
        df[id, 'vec'] = word_vec

    print("Done!")


if __name__ == '__main__':
    main()
