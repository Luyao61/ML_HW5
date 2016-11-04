#!/home/luyaoz/anaconda3/bin/python


import sys
import os
import numpy as np
import math
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB

###############################################################################


def transfer(fileDj, vocabulary):
    wordnet_lemmatizer = WordNetLemmatizer()
    raw = fileDj.read()
    BOWDj = [0]*15
    for token in raw.split( ):
    #for token in nltk.word_tokenize(raw):
        token_stemed = wordnet_lemmatizer.lemmatize(token)
        if token_stemed in vocabulary:
            index = vocabulary.index(token_stemed) + 1 # add 1 because index if saved for 'UNKNOWN'
        else:
            index = 0
        BOWDj[index] = BOWDj[index] + 1
    return BOWDj


def loadData(Path):
    dict_predefined  = ['love', 'wonderful', 'best', 'great', 'superb',
                        'still', 'beautiful', 'bad', 'worst', 'stupid', 'waste',
                        'boring', '?', '!']
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    train_path = Path + '/training_set'
    test_path = Path + '/test_set'
    #load train set neg files
    for file in os.listdir( train_path+'/neg' ):
        f= open( train_path+'/neg' + '/' + file, 'rU')
        fearture_vector = transfer(f, dict_predefined)
        Xtrain.append(fearture_vector)
        ytrain.append(-1)
    #load train set pos files
    for file in os.listdir( train_path+'/pos' ):
        f= open( train_path+'/pos' + '/' + file, 'rU')
        fearture_vector = transfer(f, dict_predefined)
        Xtrain.append(fearture_vector)
        ytrain.append(1)

    #load test set neg files
    for file in os.listdir( test_path+'/neg' ):
        f= open( test_path+'/neg' + '/' + file, 'rU')
        fearture_vector = transfer(f, dict_predefined)
        Xtest.append(fearture_vector)
        ytest.append(-1)
    #load test set pos files
    for file in os.listdir( test_path+'/pos' ):
        f= open( test_path+'/pos' + '/' + file, 'rU')
        fearture_vector = transfer(f, dict_predefined)
        Xtest.append(fearture_vector)
        ytest.append(1)


    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    thetaPos = []
    thetaNeg = []
    print(len(Xtrain[0]))
    total_words_pos = 0
    total_words_neg = 0
    dict_count_pos = [0]*len(Xtrain[0])
    dict_count_neg = [0]*len(Xtrain[0])
    for a, label in zip(Xtrain, ytrain):
        for j in range(len(a)):
            if(label == 1):
                total_words_pos = total_words_pos + a[j]
                dict_count_pos[j] = dict_count_pos[j] + a[j]
            elif(label == -1):
                total_words_neg = total_words_neg + a[j]
                dict_count_neg[j] = dict_count_neg[j] + a[j]
    for i in range(len(Xtrain[0])):
        thetaPos.append((dict_count_pos[i] + 1.0)/(total_words_pos + len(Xtrain[0])) * 1.0)
        thetaNeg.append((dict_count_neg[i] + 1.0)/(total_words_neg + len(Xtrain[0])) * 1.0)

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    yPredict = []
    accurate_count = 0
    P_Pos = math.log(0.5)
    P_Neg = math.log(0.5)

    for a, label in zip(Xtest,ytest):
        P_Pos_a = 1
        P_Neg_a = 1
        for i in range(len(Xtest[0])):
            P_Pos_a = P_Pos_a + math.log(thetaPos[i]) * a[i]
            P_Neg_a = P_Neg_a + math.log(thetaNeg[i]) * a[i]
        if(P_Pos_a > P_Neg_a):
            yPredict.append(1)
            if(label == 1): accurate_count = accurate_count + 1
        else:
            yPredict.append(-1)
            if(label == -1): accurate_count = accurate_count + 1
    Accuracy = accurate_count/len(Xtest) * 1.0
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)
    Accuracy = clf.score(Xtest,ytest)
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
