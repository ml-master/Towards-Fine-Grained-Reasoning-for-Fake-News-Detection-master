#### Bearer Token
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJtRNQEAAAAA%2BsvNNZ%2FJfisYNXG22znSrqinopc%3DUVxfo1n4Rbz5LkttdPcn1GzFEPwDVSkVcflbPvV0AhMm9pB5EH"

#### API Key
CONSUMER_KEY = "1pkpOHWb8cQzRNmUlx48ivmks"

#### API Secret Key
CONSUMER_SECRET = "6mgaMYkRKZe1SPbpXswJyMajIw8m8z7dTYLM2A7jKJ1gwVigrr"

#### Access Token
ACCESS_TOKEN = "1179274420239495168-Yt4ndjK1chRTpjbOnmoTkHtpTtow1n"

#### Access Token Secret
ACCESS_TOKEN_SECRET = "dutUnxVujwbLWocL253QDodlcyun2vi7ZEv1dJPqmDXzl"

TWEEPY_MAX_BATCH = 100

FAKE_NEWS_DATA = "C:\\Workspace\\FakeNews\\fake_news_data"
POLITIFACT_NAME = "politifact"
GOSSIPCOP_NAME = "gossipcop"
POLITIFACT_FAKE_NAME = "politifact_fake"
POLITIFACT_REAL_NAME = "politifact_real"
GOSSIPCOP_FAKE_NAME = "gossipcop_fake"
GOSSIPCOP_REAL_NAME = "gossipcop_real"

POLITIFACT_FAKE = FAKE_NEWS_DATA + "\\" + POLITIFACT_FAKE_NAME
POLITIFACT_REAL = FAKE_NEWS_DATA + "\\" + POLITIFACT_REAL_NAME
GOSSIPCOP_FAKE = FAKE_NEWS_DATA + "\\" + GOSSIPCOP_FAKE_NAME
GOSSIPCOP_REAL = FAKE_NEWS_DATA + "\\" + GOSSIPCOP_REAL_NAME

POLITIFACT_DATASET_NAMES = [POLITIFACT_FAKE_NAME, POLITIFACT_REAL_NAME]
GOSSIPCOP_DATASET_NAMES = [GOSSIPCOP_FAKE_NAME, GOSSIPCOP_REAL_NAME]

DATASET_NAMES = {
    GOSSIPCOP_NAME: GOSSIPCOP_DATASET_NAMES,
    POLITIFACT_NAME: POLITIFACT_DATASET_NAMES
}

KEYS = ["id", "name", "screen_name", 'location', 'description', \
            'followers_count', 'friends_count', 'listed_count',\
            'created_at', 'favourites_count', 'geo_enabled', 'verified', 'statuses_count' ]

KEYS_STR = ["name", "screen_name", 'description']
KEYS_LOC = ['location']

USER_IDENTITY_KEYS = ['is_tweeter', 'is_retweeter', 'is_replier']

label_d = {
    POLITIFACT_FAKE_NAME: 1,
    POLITIFACT_REAL_NAME: 0,
    GOSSIPCOP_FAKE_NAME : 1,
    GOSSIPCOP_REAL_NAME : 0
}

TYPE_TEXT = {
    'tweet': 0,
    'retweet': 1,
    'reply': 2,
}

# config.TYPE_TWEET = 0
# config.TYPE_RETWEET = 1
# config.TYPE_COMMENT = 2
TYPE_TWEET = 0
TYPE_RETWEET = 1
TYPE_COMMENT = 2

NODE_TWEET = 0
NODE_KEYWORD = 1
NODE_USER = 2

TYPE_EXAMPLE = {
    "no_tweets": 1,
    "no_replies": 1,
}

SEED = 21

TEXT_STATUS_KEYS = ["has_news_article", "has_tweets", "has_retweets", "has_replies"]

TRUE = 1
FALSE = 0
EMPTY_STR = ""
INVALID_TIMESTAMP = 0


# Additional args, for GCAN ONLY
vocab_size = 20000
retweet_user_size = 40
vocab_size = 20000
GCN_output_dim = 32
n_features = 12

source_tweet_output_dim = 32
source_tweet_length = 30
number_of_feature = 10
cnn_output_dim = 32
cnn_output_length = 38
filter_size = 3
co_attention_output_dim = 64
output_dim = 32

