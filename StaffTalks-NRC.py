from collections import OrderedDict, defaultdict, Counter
import pandas as pd
import csv
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer




sts = pd.read_csv('/home/mamonu/PycharmProjects/pyVader/Q8.csv',header=None, sep=';')

stafftalks= sts[0]


wordList = defaultdict(list)
emotionList = defaultdict(list)
with open('/home/mamonu/PycharmProjects/pyVader/NRC-emotion-lexicon-v0.92.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t') # the lexicon is tab-delimited.
    headerRows = [i for i in range(0, 46)]
    for row in headerRows:
        next(reader)
    for word, emotion, present in reader:
        if int(present) == 1:
            #print(word)   #see what we found
            wordList[word].append(emotion)
            emotionList[emotion].append(word)



tt = TweetTokenizer()
def generate_emotion_count(string, tokenizer):
    emoCount = Counter()
    for token in tt.tokenize(string):
        token = token.lower()
        emoCount += Counter(wordList[token])
    return emoCount




emotionCounts = [generate_emotion_count(tweet, tt) for tweet in stafftalks]

emotion_df = pd.DataFrame(emotionCounts, index=stafftalks.index)
emotion_df = emotion_df.fillna(0)


emotion_df['NRCvalence'] = emotion_df['positive'] - emotion_df['negative']

print emotion_df['NRCvalence']








#Combine all text together and form a Named Entity Dictionary
alltext = ' '.join([i for i in sts[0]])
print alltext
tokens = nltk.word_tokenize(alltext)
pos_tags = nltk.pos_tag(tokens)
chunked_nes = nltk.ne_chunk(pos_tags, binary=True)
nes = [' '.join(map(lambda x: x[0], ne.leaves())) for ne in chunked_nes if isinstance(ne, nltk.tree.Tree)]
ne_vocabulary = list(set(nes))

print (ne_vocabulary)

