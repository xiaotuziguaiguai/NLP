#https://www.jianshu.com/p/dfcc59c4ce8c
##################################
#Spark MLLib情感分析
#Apach Spark
#安装pyspark
$ wget http://labfile.oss.aliyuncs.com/courses/722/spark-2.0.2-bin-hadoop2.7.tgz
$ tar zxvf spark-2.0.2-bin-hadoop2.7.tgz
$ sudo mv spark-2.0.2-bin-hadoop2.7 /usr/bin/spark-2.0.2
#进入目录/usr/bin/spark-2.0.2/conf中设置配置文件spark-env.sh
$ cd /usr/bin/spark-2.0.2/conf

# 使用模板文件复制一份配置文件进行设置
$ sudo cp spark-env.sh.template spark-env.sh
$ vim spark-env.sh #编辑spark-env.sh，所使用的编辑器可根据个人爱好

# 使用vim或gedit编辑文件spark-env.sh
# 添加以下内容设置spark的环境变量
export SPARK_HOME=/usr/bin/spark-2.0.2

wget http://labfile.oss.aliyuncs.com/courses/722/numpy-1.11.3-cp27-cp27mu-manylinux1_x86_64.whl
wget http://labfile.oss.aliyuncs.com/courses/722/pyparsing-2.1.10-py2.py3-none-any.whl
wget http://labfile.oss.aliyuncs.com/courses/722/pytz-2016.10-py2.py3-none-any.whl
wget http://labfile.oss.aliyuncs.com/courses/722/cycler-0.10.0-py2.py3-none-any.whl
wget http://labfile.oss.aliyuncs.com/courses/722/python_dateutil-2.6.0-py2.py3-none-any.whl
wget http://labfile.oss.aliyuncs.com/courses/722/matplotlib-1.5.3-cp27-cp27mu-manylinux1_x86_64.whl

sudo pip install numpy-1.11.3-cp27-cp27mu-manylinux1_x86_64.whl
sudo pip install pyparsing-2.1.10-py2.py3-none-any.whl
sudo pip install pytz-2016.10-py2.py3-none-any.whl
sudo pip install cycler-0.10.0-py2.py3-none-any.whl
sudo pip install python_dateutil-2.6.0-py2.py3-none-any.whl
sudo pip install matplotlib-1.5.3-cp27-cp27mu-manylinux1_x86_64.whl
#可视化
$ wget http://labfile.oss.aliyuncs.com/courses/722/basemap-1.0.7.tar.gz
$ tar zxvf basemap-1.0.7.tar.gz
$ cd basemap-1.0.7
$ cd geos-3.3.3
$ ./configure
$ make
$ sudo make install
$ cd ..
# 先更新依赖包,再进行安装
$ sudo apt-get build-dep python-lxml
$ sudo python setup.py install

# 安装后更新，并安装python-tk
$ sudo apt-get update
$ sudo apt-get install python-tk
$ cd examples
$ python simpletest.py
#进入Code目录中并创建 shiyanlou_cs722 目录，通过以下命令获得tweets.json, donald.json及hillary.json文件。
$ wget http://labfile.oss.aliyuncs.com/courses/722/tweets.json
$ wget http://labfile.oss.aliyuncs.com/courses/722/donald.json
$ wget http://labfile.oss.aliyuncs.com/courses/722/hillary.json


# -*- coding=utf8 -*- 
from __future__ import print_function
import json
import re
import string
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName("sentiment_analysis")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
sqlContext = SQLContext(sc)
#寻找推文的协调性
#符号化推文的文本
#删除停用词，标点符号，url等
remove_spl_char_regex = re.compile('[%s]' % re.escape(string.punctuation))  # regex to remove special characters
stopwords = [u'rt', u're', u'i', u'me', u'my', u'myself', u'we', u'our',               u'ours', u'ourselves', u'you', u'your',
             u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers',
             u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what',
             u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were',
             u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a',
             u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by',
             u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after',
             u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under',
             u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all',
             u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not',
             u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don',
             u'should', u'now']

# tokenize函数对tweets内容进行分词
def tokenize(text):
    tokens = []
    text = text.encode('ascii', 'ignore')  # to decode
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                  text)  # to replace url with ''
    text = remove_spl_char_regex.sub(" ", text)  # Remove special characters
    text = text.lower()

    for word in text.split():
        if word not in stopwords \
                and word not in string.punctuation \
                and len(word) > 1 \
                and word != '``':
            tokens.append(word)
    return tokens











