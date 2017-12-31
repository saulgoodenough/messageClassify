# -*-coding:utf-8-*-

import jieba
import json
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
'''
此文件用于处理数据
'''


def cutToWords(str, stop_word):
    # 去停用词后，返回格式为空格隔开各个词
    x = jieba.lcut(str, cut_all=False)
    y = x
    for word in x:
        if word in stop_word:
            y.remove(word)
    return ' '.join(y)

t0 = time()
# with open('test.txt', 'r') as file_object:
#     for line in file_object.readlines():
#         print(line[0])
#         print(line[2:-1])
print '切词并计算短信的tf-idf向量'
#str = '商业秘密的秘密性那是维系其商业价值和垄断地位的前提条件之一'
stop_word = [line.strip().decode('utf-8') for line in open('stopwords.txt').readlines()]  # 停用词表
#print cutToWords(str, stop_word)

corpus = []
label_list = []
num_sample = 0
with open(u'带标签短信.txt', 'r') as file_object: #u'带标签短信.txt'
     for line in file_object.readlines():
         num_sample = num_sample + 1
         label_list.append(int(line[0]))
         str = line[2:-1]
         wordStr = cutToWords(str, stop_word)
         corpus.append(wordStr)

vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
#weight=tfidf.toarray() #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
### 这里不能直接全部使用toarray 因为所占内存太大
#print tfidf[0][:]
## 主成分分析
print("提取tfidf耗费时间是 %fs" % (time() - t0))
print("Performing dimensionality reduction using LSA")
t0 = time()
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(n_components=100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(tfidf)
(xlen, ylen) = X.shape
train_data_amount = int(round((4.0/5.0)*xlen)) #训练数据量
test_data_amount = xlen - train_data_amount #测试数据量
print xlen, ylen
label_data = np.array(label_list)
train_data = X[0:(train_data_amount-1)][:]
np.save('train_data.npy', train_data)
train_label = label_data[0:(train_data_amount-1)]
np.save('train_label.npy', train_label)
test_data = X[train_data_amount:-1][:]
np.save('test_data.npy', test_data)
test_label = label_data[train_data_amount:-1]
np.save('test_label.npy', test_label)
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))
print("降维耗费时间是 %fs" % (time() - t0))


