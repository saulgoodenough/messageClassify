# -*-coding:utf-8-*-


from time import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

'''
这个程序用于多层感知机分类器的训练
'''



print '多层感知机模型进行分类'
t0  =time()
train_label = np.load('train_label.npy')
train_data = np.load('train_data.npy')
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(8, 5), random_state=1)
clf.fit(train_data, train_label)
joblib.dump(clf, 'nnp_model_1.m')
print("训练花费时间 %fs" % (time() - t0))