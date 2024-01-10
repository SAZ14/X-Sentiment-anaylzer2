#import regex
import re
import csv
import pprint
import nltk.classify
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()


fr = open('data/sampleTweetFeatureList.txt','w')

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end
prTweet = open('processedTweet.txt','w')
#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Convert $STOCK to AT_STOCK
    tweet = re.sub('\$[^\s]+','AT_STOCK',tweet)  
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    #tweet = " ".join([ stemmer.stem(kw) for kw in tweet.split(" ")])
    prTweet.write(tweet)
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')
    stopWords.append('AT_STOCK')
    fp = open(stopWordListFileName, 'rU')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
	    fr.write(w.lower() + '\n')
	    featureList_tmp.append(w.lower())
    return featureVector    
#end


def uniq(list):
    l = []
    for e in list:
        if e not in l:
            l.append(e)
    return l
#end

#start getFeatureList
def getFeatureList(fileName):
    fp = open(fileName, 'rU')
    line = fp.readline()
    featureList = []
    while line:
        line = line.strip()
        featureList.append(line)
        line = fp.readline()
    fp.close()
    return featureList
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


#Read the tweets one by one and process it
inpTweets = open('data/sampleTweets.txt', 'rU')
stopWords = getStopWordList('data/feature_list/stopwords.txt')
#featureList = getFeatureList('data/sampleTweetFeatureList.txt')
count = 0;
featureList = []
featureList_tmp = []
tweets = []
line = inpTweets.readline()
while line:
    row = line.split('@@')
    sentiment = row[1]
    tweet = row[0]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    tweets.append((featureVector, sentiment));
    line = inpTweets.readline()
#end loop
featureList = sorted(uniq(featureList_tmp))

training_set = nltk.classify.util.apply_features(extract_features, tweets)
#pp.pprint(training_set)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
testFile = 'data/testTweets.txt'
resultFile = 'data/results.txt'
rp = open(resultFile, 'w')
tp = open(testFile,'rU')
newTweet = tp.readline()
pos_tp = 0.00
pos_fn = 0.00
pos_tn = 0.00
pos_fp = 0.00
neg_tp = 0.00
neg_fn = 0.00
neg_tn = 0.00
neg_fp = 0.00
mix_tp = 0.00
mix_fn = 0.00
mix_tn = 0.00
mix_fp = 0.00
neut_tp = 0.00
neut_fn = 0.00
neut_tn = 0.00
neut_fp = 0.00
corrCount = 0.00
totCount = 0.00
while newTweet:
    testTweet = newTweet.split('@@')[0]
    origSentiment = newTweet.split('@@')[1][:-1]
    processedTestTweet = processTweet(testTweet)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
    row =  testTweet   + " @@" + origSentiment + " @@" + sentiment
    rp.write(row)

    origSentiment = origSentiment.strip()
    sentiment = sentiment.strip()

    '''print origSentiment
    print sentiment'''
    #analysis for positive class
    if(origSentiment == 'positive' and sentiment == 'positive'):
        pos_tp = pos_tp + 1
        corrCount = corrCount + 1
    if(origSentiment != 'positive' and sentiment == 'positive'):
        pos_fp = pos_fp + 1
    if(origSentiment == 'positive' and sentiment != 'positive'):
        pos_fn = pos_fn + 1

    #analysis for negative class
    if(origSentiment == 'negative' and sentiment == 'negative'):
        neg_tp = neg_tp + 1
        corrCount = corrCount + 1
    if(origSentiment != 'negative' and sentiment == 'negative'):
        neg_fp = neg_fp + 1
    if(origSentiment == 'negative' and sentiment != 'negative'):
        neg_fn = neg_fn + 1

    #analysis for mixed class
    if(origSentiment == 'mixed' and sentiment == 'mix'):
        mix_tp = mix_tp + 1
        corrCount = corrCount + 1
    if(origSentiment != 'mixed' and sentiment == 'mixed'):
        mix_fp = mix_fp + 1
    if(origSentiment == 'mixed' and sentiment != 'mixed'):
        mix_fn = mix_fn + 1

    #analysis for neutral class
    if(origSentiment == 'neutral' and sentiment == 'neutral'):
        neut_tp = neut_tp + 1
        corrCount = corrCount + 1
    if(origSentiment != 'neutral' and sentiment == 'neutral'):
        neut_fp = neut_fp + 1
    if(origSentiment == 'neutral' and sentiment != 'neutral'):
        neut_fn = neut_fn + 1
    
    totCount = totCount + 1
    #read new line and continue
    newTweet = tp.readline()

#print pos_tp + pos_fp
#calculate precision, recall and f-score for all the classes

if ((pos_tp + pos_fp)==0):
    #print "found postp and posfp to be zero"
    pos_p = 0.00
else:
    pos_p = (pos_tp/(pos_tp + pos_fp))*100.0
    pos_p = round(pos_p,2)

if ((pos_tp + pos_fn)==0):
    pos_r = 0.00
else:
    pos_r = (pos_tp/(pos_tp + pos_fn))*100
    pos_r = round(pos_r,2)



if ((neg_tp + neg_fp)==0):
    neg_p = 0.00
else:
    neg_p = (neg_tp/(neg_tp + neg_fp))*100
    neg_p = round(neg_p,2)

if ((neg_tp + neg_fn)==0):
    neg_r = 0.00
else:
    neg_r = (neg_tp/(neg_tp + neg_fn))*100
    neg_r = round(neg_r,2)

if ((mix_tp + mix_fp)==0):
    mix_p = 0.00
else:
    mix_p = (mix_tp/(mix_tp + mix_fp))*100
    mix_p = round(mix_p,2)

if ((mix_tp + mix_fn)==0):
    mix_r = 0.00
else:
    mix_r = (mix_tp/(mix_tp + mix_fn))*100
    mix_r = round(mix_r,2)

if ((neut_tp + neut_fp)==0):
    neut_p = 0.00
else:
    neut_p = (neut_tp/(neut_tp + neut_fp))*100
    neut_p = round(neut_p,2)

if ((neut_tp + neut_fn)==0):
    neut_r = 0.00
else:
    neut_r = (neut_tp/(neut_tp + neut_fn))*100
    neut_r = round(neut_r,2)

acc = round(((corrCount/totCount) * 100),2)




if ((pos_p + pos_r)==0):
    pos_fscore = 0.00
else:
    pos_fscore = round((2*(pos_p * pos_r)/(pos_p + pos_r)),2)


if ((neg_p + neg_r)==0):
    neg_fscore = 0.00
else:
    neg_fscore = round((2*(neg_p * neg_r)/(neg_p + neg_r)),2)


if ((mix_p + mix_r)==0):
    mix_fscore = 0.00
else:
    mix_fscore = round((2*(mix_p * mix_r)/(mix_p + mix_r)),2)

if ((neut_p + neut_r)==0):
    neut_fscore = 0.00
else:
    neut_fscore = round((2*(neut_p * neut_r)/(neut_p + neut_r)),2)


print('\t\tPrecision\tRecall\tF-Score\n')
print 'Positive\t\t' + str(pos_p) + '\t' + str(pos_r) + '\t' + str(pos_fscore)
print '\n'
print 'Negative\t\t' + str(neg_p) + '\t' + str(neg_r) + '\t' + str(neg_fscore)
print '\n'
print 'Neutral\t\t\t' + str(neut_p) + '\t' + str(neut_r) + '\t' + str(neut_fscore)
print '\n'
print 'Mixed\t\t\t' + str(mix_p) + '\t' + str(mix_r) + '\t' + str(mix_fscore)
#print 'No tweets classified as mixed, hence can not define performance'
print '\n'

print 'Accuracy :\t' + str(acc)+'\n'

'''print '\n'
print totCount
print '\n'
print corrCount'''

#print mix_tp + mix_tn + mix_fp + mix_fn

tp.close()
rp.close()