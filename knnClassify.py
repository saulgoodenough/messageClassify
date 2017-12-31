# -*-coding:utf-8-*-

import jieba
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
# requrie numpy  scipy
# 采用lda提取特征并对垃圾短信进行分类
# 这个程序用于将数据导入数据库, 后面操作在数据库中执行


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
print("Performing dimensionality reduction using LSA")
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
train_label = label_data[0:(train_data_amount-1)]
test_data = X[train_data_amount:-1][:]
test_label = label_data[train_data_amount:-1]
print 'KNN 进行分类'
## svm 分类
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_data, train_label)
predict_label = neigh.predict(test_data)
print predict_label
print test_label
precision = 0.0 + np.sum(np.absolute(predict_label - test_label))
precision =1.0- precision * 1.0 / test_data_amount
print("accuracy = %f%%" % (precision * 100))
precision_0 = 0.0
precision_1 = 0.0
precision_2 = 0.0
for i in range(0, len(test_label)):
    if test_label[i] == 0:
        precision_2 = precision_2 + 1
        if predict_label[i] == 0:
            precision_1 = precision_1 + 1
    if predict_label[i] == 0:
        precision_0 = precision_0 + 1
precision_value = precision_1/precision_0 * 100
recall_value = precision_1/precision_2 * 100
print("precision = %f%%" % (precision_value))
print("recall = %f%%" % (recall_value))
print("F1 = %f%%" % (2*(precision_value*recall_value)/(precision_value + recall_value)))


joblib.dump(neigh, 'knn_model_1.m')
print("done in %fs" % (time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))





# svd = TruncatedSVD(n_components=80, random_state=42)
# svd.fit(tfidf)
# print(svd.explained_variance_ratio_)
# print(svd.explained_variance_ratio_.sum())


## print '将特征写入文件中'
# tfidf_dict = dict()
# for i in range(0, num_sample):
#     tfidf_dict[i] = list(X[i][:])
# print tfidf_dict[0]
# with open("test.json", "w") as outfile:
#     json.dump(tfidf_dict, outfile)

print("done in %fs" % (time() - t0))