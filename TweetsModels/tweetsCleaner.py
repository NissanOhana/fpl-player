import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import xlrd
import re
import np

def preprocessing_tweets():
    nltk.download("stopwords")
    nltk.download("wordnet")
    wb = xlrd.open_workbook('pl_sky_sports_tweets.csv')
    sheet = wb.sheet_by_index(0)
    tweets = sheet.col_values(0)
    labels = sheet.col_values(9)
    labels = labels[1:]
    labels = np.array(labels)
    labels = labels.astype(int)
    tweets = tweets[1:]
    return tweets, labels


# Clean tweet from punctuation
def tweet_cleaner_regs(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)     # Remove @mentions
    tweet = re.sub(r'RT[\s]+', '', tweet)   # Remove RT
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)  # Remove links
    tweet = re.sub(r'\.', ' . ', tweet)
    tweet = re.sub(r'#', ' # ', tweet)
    return tweet


# Remove common words - Attribution and relevance nouns.
def tweet_cleaner_common_words(tweet):
    tweet_list = [element for element in tweet.split()]
    tokens_to_remove = [token for token in tweet_list if re.match(r'[^\W\d]*$', token)]
    cleaned_tweet = ' '.join(tokens_to_remove)
    out = [word for word in cleaned_tweet.split() if word.lower() not in stopwords.words('english')]
    return out


# Change verbs to one signal form.
def tweet_cleaner_normalization(tweet):
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet:
        normalize_text = lem.lemmatize(word, 'v')
        normalized_tweet.append(normalize_text)
    return normalized_tweet


def tweet_cleaner(tweets):
    for i, tweet in enumerate(tweets):
        tweets[i] = tweet_cleaner_regs(tweets[i])
        tweets[i] = tweet_cleaner_common_words(tweets[i])
        tweets[i] = tweet_cleaner_normalization(tweets[i])
