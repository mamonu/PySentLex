import nltk


from nltk import tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import csv
import numpy as np
import pandas as pd


rows_list=[]

sid = SentimentIntensityAnalyzer()

######################################################################
with open('/home/mamonu/PycharmProjects/pyVader/Q8.csv') as csvfile:
    reader = csv.DictReader(csvfile,delimiter="\n",fieldnames = ['text'])
    for row in reader:
        lines_list = tokenize.sent_tokenize(row['text'])
        print lines_list
        score,pos,neg,neu = 0.0,0.0,0.0,0.0
        for sentence in lines_list:
            ss = sid.polarity_scores(sentence)
            score = score + ss['compound']
            pos = pos + ss['pos']
            neg = neg + ss['neg']
            neu = neu + ss['neu']


            print ss, sentence, pos,neu,neg,score


        numofsents = len(lines_list)
        totalsentscompoundscore = score / numofsents
        totalsentspos = pos / numofsents
        totalsentsneu = neu / numofsents
        totalsentsneg = neg / numofsents

        print "total score: ", score, " numofsents:", numofsents, " sentcompound:",totalsentscompoundscore
        print "totalpos: ",totalsentspos," totalsentsneu: ",neu," totalsentsneg:",neg


        #agreggate everything into a dataframe row
        RecordtoAdd = {}  # initialise an empty dict

        RecordtoAdd.update({'numofsents': numofsents})
        RecordtoAdd.update({'analyzedtext': row['text']})

        RecordtoAdd.update({'sentcompound': totalsentscompoundscore })

        rows_list.append(RecordtoAdd)

    analysed = pd.DataFrame(rows_list)


pd.set_option('display.max_colwidth', -1)
print analysed

analysed.to_html('sentimentanalysis.html')








