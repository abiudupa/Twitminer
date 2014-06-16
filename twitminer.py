#Naive Bayes Classifier to classify labels into Sports or Politics

#Libraries
import re
import csv
import nltk.classify
import pickle
import os

#Clean : Remove unecessary components 
def clean(tweet):
    #lower case
    tweet = tweet.lower()
    #@username to username
    tweet = re.sub('@([^\s]+)','\1',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    ##word to word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

#Extracts stop words from pre-existing file
def extractStopWords(FileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(FileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

#Process individual vectors and return a vector 
def wordProcess(tweet, stopWords):
    processedVector = []
    words = tweet.split()
    for w in words:
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            processedVector.append(w.lower())
    return processedVector

#Acquires all the keywords from file
def getWordList(fileName):
    fp = open(fileName, 'r')
    line = fp.readline()
    wordList = []
    while line:
        line = line.strip()
        wordList.append(line)
        line = fp.readline()
    fp.close()
    return wordList

#Extracts words from tweets
def extractWord(tweet):
    tweet_words = set(tweet)
    words = {}
    for word in wordList:
        words['contains(%s)' % word] = (word in tweet_words)
    return words

#Process the tweets
inpTweets = csv.reader(open('data/training.csv', 'rb'), delimiter=',', quotechar='"')
stopWords = extractStopWords('data/stopwords.txt')
wordList = getWordList('data/wordList.txt')
tweets = []
for row in inpTweets:
    label = row[1]
    tweet = row[2]
    cleant = clean(tweet)
    processedVector = wordProcess(cleant, stopWords)
    tweets.append((processedVector, label))

#Train and classify
training_set = nltk.classify.util.apply_features(extractWord, tweets)
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)


#Cache the classifier
import os
def SaveClassifier(NBClassifier):
  fModel = open('BayesModel.pkl',"wb")
  pickle.dump(NBClassifier, fModel,1)
  fModel.close()
  #os.system("rm BayesModel.pkl.gz")
  os.system("gzip BayesModel.pkl")
SaveClassifier(NBClassifier)

def LoadClassifier( ):
  os.system("gunzip BayesModel.pkl.gz")
  fModel = open('BayesModel.pkl',"rb")
  NBClassifier = pickle.load(fModel)
  fModel.close()
  os.system("gzip BayesModel.pkl")
  return NBClassifier
classifier2=LoadClassifier()

#Initializations
differentWords=[]
seen=[]
correct=[]
twitter=[]
testT=csv.reader(open('data/test.csv', 'rb'), delimiter=',', quotechar='"')

#Test and classify
for row in testT:
    testTweet = row[2]
    correct.append(row[1])
    cleantTestTweet= clean(testTweet)
    twitter.append(cleantTestTweet)
    label = classifier2.classify(extractWord(wordProcess(cleantTestTweet, stopWords)))
    seen.append(label)
    #print("label = %s\n" % (label)) #prints the label(Sports or Politics)
    
#Accuracy
count=0
for i in range(0,499):
        if(seen[i]==correct[i]):
           count=count+1;
        else:
           differentWords.append(twitter[i]) 
print((count*100)/(500.0))
        

