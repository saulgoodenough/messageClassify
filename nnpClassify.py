# -*-coding:utf-8-*-

import numpy as np
from sklearn.externals import joblib
from time import time

t0 = time()
clf = joblib.load('nnp_model_1.m')
test_label = np.load('test_label.npy')
test_data = np.load('test_data.npy')
test_data_amount = len(test_label)
predict_label = clf.predict(test_data)
print predict_label
print test_label
precision = 0.0 + np.sum(np.absolute(predict_label - test_label))
precision =1.0 - precision * 1.0 / test_data_amount
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
print("测试花费时间 %fs" % (time() - t0))