# -*-coding:utf-8-*-

from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib
from time import time


print 'SVM 进行分类'
## svm 分类
t0  =time()
train_label = np.load('train_label.npy')
train_data = np.load('train_data.npy')
clf = SVC()
clf.fit(train_data, train_label)
joblib.dump(clf, 'svm_model_1.m')
print("训练时间是 %fs" % (time() - t0))