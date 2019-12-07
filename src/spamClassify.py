import os
from collections import Counter
from tqdm import tqdm

import jieba.analyse as analyse
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from spam.extraFunction import spamBayes 


# 加载自定义对象，以便使用定义好的函数
spam = spamBayes()

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

# 提取训练集里所有邮件的路径
print("开始提取训练集邮件路径")
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
wordNum = 1800
wordDictList = Counter(wordTable) # 统计列表中的词频，并对应成字典
wordDictList = wordDictList.most_common(wordNum) # 取字典中词频最高的N项

# 构造训练集的特征矩阵
print("开始构造训练集的特征矩阵")
trainData = np.zeros((len(trainFileList), wordNum))
for mailIndex, trainEmail in enumerate(tqdm(trainFileList)):
    mail2Str = open(trainEmail, encoding='gb18030').read()
    # 在每个邮件中使用tf-idf提取50个关键词，统计词频并对应生成D维特征
    keyWords = analyse.extract_tags(mail2Str, topK=50, withWeight=False)
    for keyWord in keyWords:
        for featureIndex, word in enumerate(wordDictList):
            if word[0] == keyWord:
                trainData[mailIndex][featureIndex] = keyWords.count(keyWord)

# 生成训练集对应的标签
f = open(index_path)
print(f.readline()) # 舍弃第一行
train_label = np.zeros(len(trainFileList))
line = f.readline()
fileIndex = 0
while line:
    label, secondPath = line.split(" ")
    if(os.path.exists(os.path.join(train_path[:-5], secondPath[3:]))):
        if label == 'ham':
            train_label[fileIndex] = 1
        elif label == 'spam':
            train_label[fileIndex] = 0
        fileIndex += 1
    line = f.readline()
f.close()

# 保存训练特征和训练标签
np.save(r'../save', trainData)
np.save(r'../save', train_label)

# 加载和训练朴素贝叶斯分类器
naiveBayes = MultinomialNB()
naiveBayes.fit(trainData, train_label)

# 获取测试集里所有文件的路径列表
print("正在获取测试集文件路径")
testFileList = []
for path in os.listdir(test_path):
    secondPathList = os.listdir(os.path.join(test_path, path))
    for secondPath in secondPathList:
        testFileList.append(os.path.join(test_path, path, secondPath))

# 构建测试集的特征矩阵
print("开始构造训练集的特征矩阵")
testData = np.zeros((len(testFileList), wordNum))
for mailIndex, testEmail in enumerate(tqdm(testFileList)):
    mail2Str = open(testEmail, encoding='gb18030').read()
    # 在每个邮件中使用tf-idf提取50个关键词，统计词频并对应生成D维特征
    keyWords = analyse.extract_tags(mail2Str, topK=50, withWeight=False)
    for keyWord in keyWords:
        for featureIndex, word in enumerate(wordDictList):
            if word[0] == keyWord:
                trainData[mailIndex][featureIndex] = keyWords.count(keyWord)

# 预测
predict = naiveBayes.predict(testData)
np.save(r'../save', predict)

# 输出result.txt文件
print("正在写入结果文件")
f = open(r"./result.txt", 'w')
f.write("TYPE ID\n")
for index, testPath in enumerate(tqdm(testFileList)):
    firstPath, secondPath = testPath[-7:].split("\\")
    if predict[index] == 1:
        f.write("ham ../Data/" + firstPath + '/' + secondPath + '\n')
    elif predict[index] == 0:
        f.write("spam ../Data/" + firstPath + '/' + secondPath + '\n')
f.close()