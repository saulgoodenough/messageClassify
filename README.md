### LSA, SVM和多层感知机用于垃圾短信分类

环境: python2.7 

依赖的包: sklearn  numpy  jieba

database.py 用于处理数据，包括提取tf-idf向量和使用LSA进行数据的降维, 降维到100维,train_data.npy, train_label.npy, test_data.npy, test_label.npy都是保存的处理好的数据；nnp_train.py 是多层感知机的训练模型，感知机隐含层两层，分别为8和5个节点，训练完后保存模型为nnp_model_1.m, nnpClassify.py是在测试集上测试多层感知机分类模型；stopwords.txt为中文停用词集；svm_train.py是svm的训练模型，C取1，训练完后保存模型为svm_model_1.m, svmClassify.py是svm分类模型在测试集上的测试.

程序的运行顺序是：先运dataBase.py, nnp_train.py, nnpClassify.py, svm_train.py,svmClassify.py以查看两种方法的训练效果和测试效果.

项目地址:[LSA, SVM和多层感知机用于垃圾短信分类]()
