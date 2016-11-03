#!/home/luyaoz/anaconda3/bin/python


import sys
import os
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
from sklearn.naive_bayes import MultinomialNB

###############################################################################


def transfer(fileDj, vocabulary):

    return BOWDj


def loadData(Path):
    dict_predefined  = ['UNKNOWN','love', 'wonderful', 'best', 'great', 'superb',
                        'still', 'beautiful', 'bad', 'worst', 'stupid', 'waste',
                        'boring', '?', '!']
    Xtrain = np.array([1400,15])
    ytrain = np.array([1400,1])
    Xtest = np.array([600,15])
    ytest = np.array([600,1])
    train_path = Path + '/training_set'
    test_path = Path + '/test_set'
    #load train set neg files

    for file in os.listdir( train_path+'/neg' ):
        f= open( train_path+'/neg' + '/' + file, 'rU')
        raw = f.read()
        for tokens in nltk.word_tokenize(raw):
            print(tokens)

 from nltk.stem import WordNetLemmatizer
>>> wordnet_lemmatizer = WordNetLemmatizer()
>>> wordnet_lemmatizer.lemmatize(‘dogs’)

    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):

    return Accuracy



def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg):

    return yPredict


def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg):
    yPredict = []

    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):

    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []

    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    print ("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]


    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print ("thetaPos =", thetaPos)
    print ("thetaNeg =", thetaNeg)
    print ("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print ("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print ("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    print ("Directly MNBC tesing accuracy =", Accuracy)
    print ("--------------------")

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print ("thetaPosTrue =", thetaPosTrue)
    print ("thetaNegTrue =", thetaNegTrue)
    print ("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print ("BNBC classification accuracy =", Accuracy)
    print ("--------------------")
