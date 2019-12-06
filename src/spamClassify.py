import os
from collections import Counter
from tqdm import tqdm

import jieba.analyse as analyse
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from spam.extraFunction import spamBayes 


# 加载自定义对象，以便使用定义好的函数
spam = spamBayes();

# 文件路径
train_path = r'..\data\train\Data'
test_path = r'..\data\test\Data'
index_path = r'..\data\index.txt'

# 排除一些不必要统计的单词
exclude_words = ['DOC','DOCNO','FROM','TO','SUBJECT','TEXT']
# 获取中文停用词并与英文词合并
stopList = spam.getStopWords() + exclude_words 

trainFileList = [] # 存储训练集邮件路径
wordTable = [] # 存放整个训练集中提出来的所有主题词
i = 1

# 提取训练集里所有邮件的路径
for path in os.listdir(train_path):
    secondPathList = os.listdir(os.path.join(train_path, path))
    for secondPath in secondPathList:
        trainFileList.append(os.path.join(train_path, path, secondPath))

# 提取训练集的关键词
print("开始提取训练集的关键词")
for trainEmail in tqdm(trainFileList):
    # 读入整个邮件为str
    mail2Str = open(trainEmail, encoding='gb18030').read()
    # 每个邮件中提取50个主题词，使用TF-IDF算法
    keyWords = analyse.extract_tags(mail2Str, topK=50, withWeight=False)
    for keyWord in keyWords:
        if keyWord not in stopList:
            wordTable.append(keyWord)

# 提取频率最高的1200个词汇
wordNum = 1200
wordDict = Counter(wordTable) # 统计列表中的词频，并对应成字典
wordDict = wordDict.most_common(wordNum) # 取字典中词频最高的N项

# 构造训练集的特征矩阵
print("开始构造训练集的特征矩阵")
train_matrix = np.zeros((len(trainFileList), wordNum))
mailIndex = 0

for trainEmail in tqdm(trainFileList):
    mail2Str = open(trainEmail, encoding='gb18030').read()
    keyWords = analyse.extract_tags(mail2Str, topK=50, withWeight=False)
    for keyWord in keyWords:
        wordId = 0
        for index, word in enumerate(wordDict):
            if word[0] == keyWord:
                wordId = index
                train_matrix[mailIndex][wordId] = keyWords.count(keyWord)
    mailIndex += 1 

# 制作训练样本的标签集合
f = open(index_path)
print(f.readline)#next(f)
train_label = np.zeros(len(trainFileList))
i = 0
for label in train_label:
    s = f.readline()
    if s[:3] == 'ham':
        train_label[i] = 1
    i += 1

# 加载朴素贝叶斯分类器
naiveBayes = MultinomialNB()
# 保存训练特征和训练标签
np.save(r'C:\Users\Administrator\Desktop\SpamClassfier\MySpamBayes\classify1.py\train_matrix', train_matrix)
np.save(r'C:\Users\Administrator\Desktop\SpamClassfier\MySpamBayes\classify1.py\train_label', train_label)
# 训练模型
naiveBayes.fit(train_matrix, train_label)

# 得到测试数据
trainFileList.clear()
for f1 in os.listdir(test_path):
    test_dir = os.path.join(test_path, f1)
    trainFileList += [os.path.join(test_dir, f2) for f2 in os.listdir(test_dir)]

test_matrix = np.zeros((len(trainFileList), wordNum))
mailIndex = 0
for trainEmail in tqdm(trainFileList):
    try:
        f = open(trainEmail, encoding='gb18030').read()
    except UnicodeDecodeError:
        print("wrong trainEmail:" + trainEmail)
    
    keyWords = analyse.extract_tags(f, 50, withWeight = False)
    for keyWord in keyWords:
        wordId = 0
        for index, word in enumerate(wordDict):
            if keyWord == word[0]:
                wordId = index
                test_matrix[mailIndex][wordId] = keyWords.count(keyWord)
    mailIndex += 1

# 预测
predict = naiveBayes.predict(test_matrix)
np.save(r'C:\Users\Administrator\Desktop\SpamClassfier\MySpamBayes', predict)

# 获取结果
i = 0
a = open('index.txt', 'w')
a.write("TYPE ID"+"\n")
for trainEmail in trainFileList:
    trainEmail = "../Data/{0}/{1}".format(trainEmail.split("\\")[-2],trainEmail.split("\\")[-1])
    if predict[i] == 1:
        tag = 'ham {0}'.format(trainEmail)
    else:
        tag = 'spam {0}'.format(trainEmail)
    i += 1
    a.write(tag+"\n")
