from encoder import Model, sst_binary, model_train, model_predict, model_load

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


model = Model(max_length)

all_data = sst_binary(load_path)  #分别获取所有的句子和标签
print('=> Succeeds in loading <' + language + '> file and starting to translate words into Embeddedness······')

x, y, wi = model.transform(all_data)  #将每个句子里的词转化成词频索引值
print('=> Succeeds in translating swords into word Embeddedness and starting to train the model process······')

accuracy = model_train(x, y, wi, language, max_length, tr_num, va_num)  #训练模型  (如果已经有训练好的模型，这行代码注释掉)
print('=> accuracy: ', accuracy*100, '%')

# model_load(language) #如果模型训练好了，调用此方法直接加载模型，不需要再训练

while True:
    sentence = input("Please enter a single sentence to predict:")
    result = model_predict(sentence, max_length)
    if result == 0:
       print('negative')
    else:
       print("positive")
