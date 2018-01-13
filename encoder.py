import os
import html
import numpy as np
import pandas as pd
import jieba

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM


# 获取csv文件：内容放到数组里面 分别是训练集、验证集、测试集，最后合并到一起
def sst_binary(data_dir='data/chinese'):
    tr_data = pd.read_csv(os.path.join(data_dir, 'train_binary_sent.csv'))
    va_data = pd.read_csv(os.path.join(data_dir, 'valid_binary_sent.csv'))
    te_data = pd.read_csv(os.path.join(data_dir, 'test_binary_sent.csv'))
    all_data = tr_data.append(va_data).append(te_data)
    return all_data

# krea 训练数据集
def model_train(x, y, wi, language, sentence_max_length=100, tr_num=17000, va_num=2000):
    global model
    model = Sequential()
    model.add(Embedding(wi, 256, input_length=sentence_max_length))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        x[:tr_num],
        y[:tr_num],
        batch_size=128,
        nb_epoch=30,
        validation_data=(x[tr_num:tr_num + va_num], y[tr_num:tr_num + va_num]))
    score = model.evaluate(
        x[tr_num + va_num:], y[tr_num + va_num:], batch_size=128)

    model.save('model_' + language + '.h5')
    
    return score[1]


#单个句子的预测函数
def model_predict(s, sentence_max_length=100):
    s = np.array(word2num(list(jieba.cut(s)), sentence_max_length))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]


#将词转化为数字向量 即一个句子里的每个词都有用上面生成的索引值代替
def word2num(s, sentence_max_length):
    s = [i for i in s if i in word_set]
    s = s[:sentence_max_length] + [''] * max(0, sentence_max_length - len(s))
    return list(word_frequency[s])


#定义模型
class Model(object):
    def __init__(self, sentence_max_length=100):

        sentence_max_length = sentence_max_length  #截断词数 cut texts after this number of words (among top max_features most common words)
        sentence_drop_length = 5  #出现次数少于该值的词扔掉。这是最简单的降维方法

        #将每个句子里的词转化成词频索引值
        def transform(data):

            #如果是中文调用结巴分词
            xs = data['sentence'].apply(lambda s: list(jieba.cut(s)))

            #将所有词放到一个数组中
            word_all = []
            for i in xs:
                word_all.extend(i)

            #统计词频并排序建索引
            global word_frequency, word_set
            word_frequency = pd.Series(word_all).value_counts()  #统计词频，从大到小排序
            word_frequency = word_frequency[word_frequency >=
                                            sentence_drop_length]  #出现次数小于5的丢弃
            word_frequency[:] = list(range(
                1,
                len(word_frequency) + 1))  #将词频排序的结果加索引
            word_frequency[''] = 0  #添加空字符串用来补全，之前丢弃的后面的找不到的会用0代替
            word_set = set(
                word_frequency.index)  #经过处理之后的所有词的数组集合,并且去掉可能存在的重复元素

            #将词语替换成按照所有训练集词频排序后的索引
            xt = xs.apply(lambda s: word2num(s, sentence_max_length))
            xt = np.array(list(xt))
            yt = np.array(list(data['label'])).reshape(
                (-1, 1))  #此处用来调整标签形状n行1列 (-1是模糊控制即有不定多少行，1是1列)

            #当前训练集合词的索引长度
            wi = len(word_frequency)

            return xt, yt, wi

        self.transform = transform
