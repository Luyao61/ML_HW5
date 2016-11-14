#!/home/luyaoz/anaconda3/bin/python

import sys
import os
import numpy as np
import math
from sklearn.naive_bayes import MultinomialNB

###############################################################################


def transfer(fileDj, vocabulary):
    f= open( fileDj, 'rU')
    raw = f.read()
    BOWDj = [0]* len(vocabulary)
    for token in raw.split( ):
    #for token in nltk.word_tokenize(raw):
        if token in vocabulary:
            index = vocabulary.index(token)
        elif token in word_love_inflected:
            index = vocabulary.index('love')
        else:
            index = vocabulary.index('UNK')
        BOWDj[index] = BOWDj[index] + 1
    return BOWDj


def loadData(Path):
    global dict_predefined
    global word_love_inflected
    dict_predefined  = ['love', 'wonderful', 'best', 'great', 'superb',
                        'still', 'beautiful', 'bad', 'worst', 'stupid', 'waste',
                        'boring', '?', '!', 'UNK']
    word_love_inflected = ['loving', 'loved', 'loves']

    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    train_path = Path + 'training_set'
    test_path = Path + 'test_set'
    #load train set neg files
    for file in os.listdir( train_path+'/neg' ):
        file_path = train_path+'/neg/' + file
        fearture_vector = transfer(file_path, dict_predefined)
        Xtrain.append(fearture_vector)
        ytrain.append(-1)
    #load train set pos files
    for file in os.listdir( train_path+'/pos' ):
        file_path = train_path+'/pos/' + file
        fearture_vector = transfer(file_path, dict_predefined)
        Xtrain.append(fearture_vector)
        ytrain.append(1)

    #load test set neg files
    for file in os.listdir( test_path+'/neg' ):
        file_path = test_path+'/neg/' + file
        fearture_vector = transfer(file_path, dict_predefined)
        Xtest.append(fearture_vector)
        ytest.append(-1)
    #load test set pos files
    for file in os.listdir( test_path+'/pos' ):
        file_path = test_path+'/pos/' + file
        fearture_vector = transfer(file_path, dict_predefined)
        Xtest.append(fearture_vector)
        ytest.append(1)


    return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
    thetaPos = []
    thetaNeg = []
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
        P_Pos_a = P_Pos
        P_Neg_a = P_Neg
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



def naiveBayesMulFeature_testDirectOne(path, thetaPos, thetaNeg, vocabulary):

    P_Pos = math.log(0.5)
    P_Neg = math.log(0.5)

    f= open( path, 'rU')
    raw = f.read()
    tokens = raw.split()
    P_Pos_file = P_Pos
    P_Neg_file = P_Neg
    for token in tokens:
        if token in vocabulary:
            index = vocabulary.index(token)
        elif token in word_love_inflected:
            index = vocabulary.index('love')
        else:
            index = vocabulary.index('UNK')
        P_Pos_file += math.log(thetaPos[index])
        P_Neg_file += math.log(thetaNeg[index])
    if(P_Pos_file > P_Neg_file):
        yPredict = 1
    else:
        yPredict = -1

    return yPredict


def naiveBayesMulFeature_testDirect(path, thetaPos, thetaNeg, vocabulary):
    yPredict = []

    accurate_count = 0
    n_files = 0.0
    for file in os.listdir( path + 'neg' ):
        file_path = path+'neg' + '/' + file
        n_files = n_files + 1
        y_ = naiveBayesMulFeature_testDirectOne(file_path, thetaPos, thetaNeg, vocabulary)
        yPredict.append(y_)
        if y_ == -1: accurate_count = accurate_count+1

    for file in os.listdir( path + 'pos' ):
        file_path = path+'pos' + '/' + file
        n_files = n_files + 1
        y_ = naiveBayesMulFeature_testDirectOne(file_path, thetaPos, thetaNeg, vocabulary)
        yPredict.append(y_)
        if y_ == 1: accurate_count = accurate_count+1

    Accuracy = accurate_count/n_files
    return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
    thetaNegTrue = [0] * len(Xtrain[0])
    thetaPosTrue = [0] * len(Xtrain[0])
    for i in range(len(Xtrain[0])):
        count = 1 # num of files which include Wi and are in Class Neg
        for j in range ( int(len(Xtrain)/2) ):
            if(Xtrain[j][i] != 0):
                count = count + 1
        # count/(num of files in class Neg + 2)
        thetaNegTrue[i] = float(count/(len(Xtrain)/2 + 2))

        count = 1 # num of files which include Wi and are in Class Pos
        for j in range ( int(len(Xtrain)/2), len(Xtrain)):
            if(Xtrain[j][i] != 0):
                count = count + 1
        # count/(num of files in class Pos + 2)
        thetaPosTrue[i] = float(count/(len(Xtrain)/2 + 2))

    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    accurate_count = 0
    for i in range(len(Xtest)):

        pos_score = 0
        neg_score = 0
        """
        pos_score = 1
        neg_score = 1
        """
        for j in range(len(Xtest[i])):
            if(Xtest[i][j] == 0 ):

                pos_score = pos_score + math.log(1-thetaPosTrue[j])
                neg_score = neg_score + math.log(1-thetaNegTrue[j])
                """
                pos_score = pos_score * (1-thetaPosTrue[j])
                neg_score = neg_score * (1-thetaNegTrue[j])
                """
            else:

                pos_score = pos_score + math.log(thetaPosTrue[j])
                neg_score = neg_score + math.log(thetaNegTrue[j])
                """
                pos_score = pos_score * thetaPosTrue[j]
                neg_score = neg_score * thetaNegTrue[j]
                """
        if(pos_score >neg_score):
            yPredict.append(1)
            if(ytest[i] == 1):
                accurate_count = accurate_count+1
        else:
            yPredict.append(-1)
            if(ytest[i] == -1):
                accurate_count = accurate_count+1

    Accuracy = float(accurate_count/len(ytest))
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

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, dict_predefined)
    print ("Directly MNBC tesing accuracy =", Accuracy)
    print ("--------------------")

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print ("thetaPosTrue =", thetaPosTrue)
    print ("thetaNegTrue =", thetaNegTrue)
    print ("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print ("BNBC classification accuracy =", Accuracy)
    print ("--------------------")
