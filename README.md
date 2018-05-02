# lstm-sentiment-analysis
sentiment analysis with chinese and english

本项目相关文章说明转到[http://slg.charmculture.com/article/5a60a16080a23b7b1c9df3f9](http://slg.charmculture.com/article/5a60a16080a23b7b1c9df3f9)



#### 情感分析工具keras，语言包括中文和英文

开始之前，你需要对深度学习原理有比较深刻的了解，lstm的原理，调参优化细节，keras基本知识的掌握。Python 版本 3.6.5

**1. 准备语料**

本次收集的语料不是太多。中文大概2w多条（淘宝评论），英文1w多条（电影评论），以后有时间会继续补充语料。其中有5%作为验证集，10%为测试集合。文件已经切分好，但是因为需要训练所有数据集，所以这个提前切分到不同文件夹并没有什么作用（后面讲到会重新合并再切分）。

**2.选择语言版本，分别设置训练集、测试集、验证集和维度**

因为后面的程序要训练中文或英文，所以在这里提前选择语言版本和不同的语言版本训练相关的参数。
```python
# 选择语言中文还是英文
languageType = ''
while (languageType != 'c' and languageType != 'e'):
    languageType = input("Please enter a train type(chinese enter lower: c , english enter lower: e): ")

max_length = ''  #一句话最大长度
load_path = ''  #文件加载路径
language = ''  #语言类型
tr_num = 17000 #训练集
va_num = 2000 #训练集

if languageType == 'c':
    max_length = 100
    load_path = 'data/chinese'
    language = 'chinese'
    tr_num = 17000
    va_num = 2000     
elif languageType == 'e':
    max_length = 40
    load_path = 'data/english'
    language = 'english'
    tr_num = 8000
    va_num = 600 

```

**3.加载数据集**
这里把中文和英文放在不同的文件夹下，利用  `pandas`中的`read_csv()`读取数据集合并到一起，如果这里本来就是一个整的数据集。则直接读取就好。

```python
# 获取csv文件：内容放到数组里面 分别是训练集、验证集、测试集，最后合并到一起
def sst_binary(data_dir='data/chinese'):
    tr_data = pd.read_csv(os.path.join(data_dir, 'train_binary_sent.csv'))
    va_data = pd.read_csv(os.path.join(data_dir, 'valid_binary_sent.csv'))
    te_data = pd.read_csv(os.path.join(data_dir, 'test_binary_sent.csv'))
    all_data = tr_data.append(va_data).append(te_data)
    return all_data
```

**4.数据预处理**

这一步是针对中文和英文的特点来处理掉对分析无用的词提升精度。比如停用词、标点符号、特殊字符、转义字符等等。因为语料比较少，这个程序还没有针对这一块做处理。

**5.将词语转化为向量**

这里是最核心的地方，深度学习在训练数据的时候要求输入的数据是一个向量，这样才能进行矩阵运算，也是多层感知器的输入。所以如果直接将一组句子是无法识别的。所以最重要的一步就是将词语转化为词向量，可是如何才能得到向量呢？

这里用到的是词嵌入的方法，大概步骤是：

1. 中文最小统计粒度是词，所以要先切词（`jieba`）将一句话按照词语切分开来而非字。
2. 将所有词放在一起，统计每个词出现的次数按照重大到小的排序，然后加上索引。
3. 将句子中的词语全部替换成相应的索引，这样一个句子中的每个词语就用一个数字去表示了。
4. 调用keras model第一层`Embedding()`，该层会利用词嵌入将句子数字数组转化为词向量。

需要注意的是，jieba分词虽然是分中文的，但是也可以处理英文（英文是按照空格切分的），这样可以得到比较统一的数组shape。
```python
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
```

**6.keras 训练数据集**

这一部分就交给keras处理了，具体用法可以参见keras中文文档，可以自定义一些参数，比如训练轮数、激活函数、加入验证集等等。当然核心的还是lstm了，相对于RNN，在训练长文本有更好的效果。训练完了之后可以选择保存模型。方便下次直接调用。

```python
#将词转化为数字向量 即一个句子里的每个词都有用上面生成的索引值代替
def word2num(s, sentence_max_length):
    s = [i for i in s if i in word_set]
    s = s[:sentence_max_length] + [''] * max(0, sentence_max_length - len(s))
    return list(word_frequency[s])

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


#加载已经训练好的模型
def model_load(language):
    global model
    model = load_model('model_' + language + '.h5')


```
![中文语言训练过程与结果](http://upload-images.jianshu.io/upload_images/3502567-db3bff5980a8bac6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![英文语言训练过程与结果](http://upload-images.jianshu.io/upload_images/3502567-9941d04c3670f58a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



**7.预测单个句子**

预测单个句子依然需要将这个句子分词，然后将词转化为数字，所以还是用到训练模型时用到的处理方式。

```python
#单个句子的预测函数

def model_predict(s, sentence_max_length=100):
    s = np.array(word2num(list(jieba.cut(s)), sentence_max_length))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]

```
好了，大功告成，我们已经可以直接测试训练的结果了。




