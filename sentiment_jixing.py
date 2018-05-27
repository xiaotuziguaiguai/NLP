#https://www.jianshu.com/p/4cfcf1610a73?nomobile=yes
#基于情感词典的文本情感极性分析
#情感词典及对应分数
#BosonNLP
#否定词词典
#程度副词词典
#停用词词典
#分词
#去停用词
# -*- coding:utf8 -*-

from collections import defaultdict
import os
import re
import jieba
import codecs
import sys
import chardet
import matplotlib.pyplot as plt



# php install jieba





#使用jieba 函数  对 sentence 文本进行分词

def sent2word(sentence):


#调用jieba进行分词
    segList = jieba.cut(sentence)

#分词后的结果存为segResult 为list类型
    segResult = []
    for w in segList:
        segResult.append(w)

#调用 readLines 读取停用词
    stopwords = readLines('stop_words.txt')

#如果是停用词 就不保存到newSent
    newSent = []
    for word in segResult:
        if word+'\n' in stopwords:
            continue
        else:
            newSent.append(word)
#返回newSent
    return newSent


#直接对 sentence 进行分词  不使用停用词 并返回（主要是根据word需要这个操作）
def returnsegResult(sentence):

    segResult = []
    segList = jieba.cut(sentence)

    for w in segList:
        segResult.append(w)
    return segResult


#获取 filepath 目录下的所有文件目录并返回
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    child=[]
    for allDir in pathDir:
        child.append(os.path.join('%s/%s' % (filepath, allDir)))
    return child

#读取 filename路径 的每一行数据 并返回 转换为GBK
def readLines(filename):
    fopen = open(filename, 'r')
    data=[]
    for x in fopen.readlines():
        if x.strip() != '':
                data.append(unicode(x.strip(),"GBK"))

    fopen.close()
    return data



#读取 filename路径 的每一行数据 并返回
def readLines2(filename):
    fopen = open(filename, 'r')
    data=[]
    for x in fopen.readlines():
        if x.strip() != '':
                data.append(x.strip())

    fopen.close()
    return data

#主要为情感定位  见程序文件相关代码 这里是为了速度 提取了部分代码 本来应该在classifyWords 里边  貌似对速度影响不大
def words():
    #情感词
    senList = readLines2('BosonNLP_sentiment_score.txt')
    senDict = defaultdict()
    for s in senList:
        senDict[s.split(' ')[0]] = s.split(' ')[1]
    #否定词
    notList = readLines2('notDict.txt')
    #程度副词
    degreeList = readLines2('degreeDict.txt')
    degreeDict = defaultdict()
    for d in degreeList:
        degreeDict[d.split(' ')[0]] = d.split(' ')[1]

    return senDict,notList,degreeDict




# 见文本文档  根据情感定位  获得句子相关得分
def classifyWords(wordDict,senDict,notList,degreeDict):

    senWord = defaultdict()
    notWord = defaultdict()
    degreeWord = defaultdict()
    for word in wordDict.keys():
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[wordDict[word]] = senDict[word]
        elif word in notList and word not in degreeDict.keys():
            notWord[wordDict[word]] = -1
        elif word in degreeDict.keys():
            degreeWord[wordDict[word]] = degreeDict[word]
    return senWord, notWord, degreeWord


#计算句子得分  见程序文档
def scoreSent(senWord, notWord, degreeWord, segResult):
    W = 1
    score = 0
    # 存所有情感词的位置的列表
    senLoc = senWord.keys()
    notLoc = notWord.keys()
    degreeLoc = degreeWord.keys()
    senloc = -1
    # notloc = -1
    # degreeloc = -1
    # 遍历句中所有单词segResult，i为单词绝对位置
    for i in range(0, len(segResult)):
        # 如果该词为情感词
        if i in senLoc:
            # loc为情感词位置列表的序号
            senloc += 1
            # 直接添加该情感词分数
            score += W * float(senWord[i])
            # print "score = %f" % score
            if senloc < len(senLoc) - 1:
                # 判断该情感词与下一情感词之间是否有否定词或程度副词
                # j为绝对位置
                for j in range(senLoc[senloc], senLoc[senloc + 1]):
                    # 如果有否定词
                    if j in notLoc:
                        W *= -1
                    # 如果有程度副词
                    elif j in degreeLoc:
                        W *= float(degreeWord[j])
        # i定位至下一个情感词
        if senloc < len(senLoc) - 1:
            i = senLoc[senloc + 1]
    return score


#列表 转 字典
def listToDist(wordlist):
    data={}
    for x in range(0, len(wordlist)):
        data[wordlist[x]]=x
    return data

#绘图相关  自行百度下
def runplt():
    plt.figure()
    plt.title('test')
    plt.xlabel('x')
    plt.ylabel('y')
    #这里定义了  图的长度 比如 2000条数据 就要 写 0,2000  
    plt.axis([0,1000,-10,10])
    plt.grid(True)
    return plt




#主题从这里开始 上边全是方法


#获取 test/neg 下所有文件 路径
filepwd=eachFile("test/neg")



#
scre_var=[]


#获取 本地的情感词 否定词 程度副词
words_vaule=words()

#循环 读取 filepwd  （也就是test/neg目录下所有文件全部跑一下）
for x in filepwd:
    #读目录下文件的内容
    data=readLines(x)
    #对data内容进行分词
    datafen=sent2word(data[0])
    #列表转字典
    datafen_dist=listToDist(datafen)
    #通过classifyWords函数 获取句子的 情感词 否定词 程度副词 相关分值
    data_1=classifyWords(datafen_dist,words_vaule[0],words_vaule[1],words_vaule[2])
    # 通过scoreSent 计算 最后句子得分
    data_2=scoreSent(data_1[0],data_1[1],data_1[2],returnsegResult(data[0]))
    # 将得分保存在score_var 以列表的形式
    score_var.append(data_2)
    #打印句子得分
    print data_2

#对所有句子得分进行倒序排列
score_var.sort(reverse=True)

#计算一个index 值 存 1~ 所有句子长度 以便于绘图
index=[]
for x in range(0,len(score_var)):
    index.append(x+1)

#初始化绘图
plt=runplt();
#带入参数
plt.plot(index,score_var,'r.')
#显示绘图
plt.show();
##########################################################
#English
# -*- coding: utf-8 -*-
# -*- coding: <encoding name> -*-

import numpy as np
import sys
import re
import codecs
import os
import jieba
import gensim, logging
from gensim.models import word2vec
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cross_validation import train_test_split
#from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
from sklearn.metrics import f1_score
#from bayes_opt import BayesianOptimization as BO
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as acc



def parseSent(sentence):
    seg_list = jieba.cut(sentence)
    output = ''.join(list(seg_list)) # use space to join them
    return output

def sent2word(sentence):
    """
    Segment a sentence to words
    Delete stopwords
    """
    segResult = []


    segList = jieba.cut(sentence)

    for w in segList:
        segResult.append(w)

    stopwords = readLines('stop_words.txt')
    newSent=[]
    stopwords_list=[]
    for word in segResult:
        if word in stopwords:
            # print "stopword: %s" % word
            continue
        else:
            newSent.append(word)

    #output = ' '.join(list(newSent))

    return newSent



def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    child=[]
    for allDir in pathDir:
        child.append(os.path.join('%s/%s' % (filepath, allDir)))
    return child

def readLines(filename):
    fopen = open(filename, 'r')
    data=[]
    for x in fopen.readlines():
        if x.strip() != '':
            data.append(x.strip())
    fopen.close()
    return data


def readFile(filename):
    data=[]
    for x in filename:
        fopen = open(x, 'r') 
        for eachLine in fopen:
            if eachLine.strip() != '':
                data.append(unicode(eachLine.strip(),"GBK"))
    fopen.close()
    return data



def getWordVecs(wordList):
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype = 'float')


def buildVecs(filename):
    posInput = []
    with open(filename, "rb") as txtfile:
        for lines in txtfile:
            lines = lines.split('\n')
            if lines[0] == "\r" or lines[0] == "\r\n" or lines[0] == "\r\r":
                pass
            else:

                for line in lines:            
                    line = list(jieba.cut(line))

                    resultList = getWordVecs(line)

                    # for each sentence, the mean vector of all its vectors is used to represent this sentence
                    if len(resultList) != 0:
                        resultArray = sum(np.array(resultList))/len(resultList)
                        posInput.append(resultArray)


    return posInput
# load word2vec model



#训练模型输出模型




filepwd=eachFile("test/new")

sentences=[]
for x in filepwd:
    data=readLines(x) 
    sentences.extend(sent2word(data[0]))    
    #sentences.append(data[0])

model = gensim.models.Word2Vec(sentences, min_count=1)
# outp1 = 'corpus.model.bin'
# model.save(outp1)

filepwd_pos=eachFile("test/pos")
filepwd_neg=eachFile("test\\neg")

pos_number=0
neg_number=0
posInput=[]
negInput=[]
for pos in filepwd_pos:
    pos_buildVecs=buildVecs(pos)
    posInput.extend(pos_buildVecs)
    pos_number+=1
    if pos_number == 100:
        break
for neg in filepwd_neg:
    neg_buildVecs=buildVecs(neg)
    negInput.extend(neg_buildVecs)
    neg_number+=1
    if neg_number == 100:
        break


y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))


X = posInput[:]

for neg in negInput:
    X.append(neg)

X = np.array(X)


X=scale(X)

X_reduced = PCA(n_components = 100).fit_transform(X)





#X_reduced_train,X_reduced_test, y_reduced_train, y_reduced_test =train_test_split(X_reduced,y)


X_reduced_train,X_reduced_test, y_reduced_train, y_reduced_test =train_test_split(X_reduced,y,test_size=0.4, random_state=1)

"""
SVM (RBF)
    using training data with 100 dimensions
"""



clf = SVC(C = 2, probability = True)
clf.fit(X_reduced_train, y_reduced_train)
print 'Test Accuracy: %.2f'% clf.score(X_reduced_test, y_reduced_test)

pred_probas = clf.predict_proba(X_reduced_test)[:,1]

#print "KS value: %f" % KSmetric(y_reduced_test, pred_probas)[0]

#plot ROC curve# AUC = 0.92# KS = 0.7


#输出相关结果 以及绘图
print "test:"
print clf.predict(X_reduced_test)
print "value:"
print y_reduced_test

test_value=clf.predict(X_reduced_test)

index=[]
for x in range(0,len(test_value)):
    index.append(x+1)

test_value_1=0
test_value_0=0
for test_value_data in test_value:
    if test_value_data == 1:
        test_value_1+=1
    else:
        test_value_0+=1

y_reduced_test_1=0
y_reduced_test_0=0
for y_reduced_test_data in y_reduced_test:
    if y_reduced_test_data == 1:
        y_reduced_test_1+=1
    else:
        y_reduced_test_0+=1

test_value_label='test pos: '+str(test_value_1)+' neg: '+str(test_value_0)
y_reduced_test_label='value pos: '+str(y_reduced_test_1)+' neg: '+str(y_reduced_test_0)


plt.plot(index, test_value,'ro',label = test_value_label)
plt.plot(index,y_reduced_test, 'b.',label =y_reduced_test_label)
plt.xlim([0, len(test_value)])
plt.ylim([-2, 2])
plt.legend(loc = 'lower right')
plt.show()











fpr,tpr,_ = roc_curve(y_reduced_test, pred_probas)
roc_auc =  sklearn.metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'roc_auc = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.show()
