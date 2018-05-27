#如何使用python作词云
#https://www.jianshu.com/p/d50a14541d01
sudo pip install snownlp 
sudo install -U textblob
python -m textblob.download_corpora
1、TextBlob
text = "I am happy today. I feel sad today."
from textblob import TextBlob 
blob = TextBlob(text)
blob
blob.sentences
blob.sentences[0].sentiment
blob.sentences[1].sentiment
blob.sentiment
2、SnowNLP
text = u"我今天很快乐。我今天很愤怒。"
from snownlp import SnowNLP 
s = SnowNLP()
for sentence in s.sentences:
  print snetence
s1 = SnowNLP(S.sentences[0])
s1.sentiments
s2 = SnowNLP(s.sentences[1])
s2.sentiments

