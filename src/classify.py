#encoding=utf-8

from spam.spamEmail import spamEmailBayes
import re
import os
from tqdm import tqdm 

#spam类对象
spam=spamEmailBayes()

#保存词频的词典
spamDict={}
normDict={}
testDict={}
#保存每封邮件中出现的词
wordsList=[]
wordsDict={}

#获得训练集里正常邮件、垃圾邮件路径列表
print("正在获得文件路径列表")
hamFileList = []
spamFileList = []
basePath = "E:\\dataset\\ir2019\\garbageClassification\\"
f1 = open(basePath + "index.txt")
f1.readline() # 舍弃第一行
line = f1.readline()
while line:
    label, secondPath = line.split(" ")
    if label == "ham":
        hamFileList.append(basePath + "train" + secondPath[2:])
    elif label == "spam":
        spamFileList.append(basePath + "train" + secondPath[2:])
    line = f1.readline()
f1.close()

# 获取测试集里所有文件的路径列表
testFileList = []
pathList = os.listdir(basePath + "test\\Data\\")
for path in pathList:
    secondPathList = os.listdir(basePath + "test\\Data\\" + path)
    for secondPath in secondPathList:
        testFileList.append(basePath + "test\\Data\\" + path + "\\" + secondPath)

# 获取训练集中正常邮件与垃圾邮件的数量
normFilelen=len(hamFileList)
spamFilelen=len(spamFileList)

# 获得停用词表，用于对停用词过滤
stopList=spam.getStopWords()

# 获得正常邮件中的词频
print("正在获得正常邮件中的词频")
for fileName in tqdm(hamFileList):
    wordsList.clear()
    try:
        for line in open(fileName[:-1], encoding='gb18030'):
            # 过滤掉非中文字符
            rule=re.compile(r"[^\u4e00-\u9fa5]")
            line=rule.sub("", line)
            # 将每封邮件出现的词保存在wordsList中 
            spam.get_word_list(line, wordsList, stopList)
    except FileNotFoundError:
        normFilelen -= 1

    # 统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
normDict = wordsDict.copy()
# print(normDict)

#获得垃圾邮件中的词频
print("正在获得垃圾邮件中的词频")
wordsDict.clear()
for fileName in tqdm(spamFileList):
    wordsList.clear()
    try:
        for line in open(fileName[:-1], encoding='gb18030'):
            # 过滤掉非中文字符
            rule=re.compile(r"[^\u4e00-\u9fa5]")
            line=rule.sub("",line)
            # 将每封邮件出现的词保存在wordsList中
            spam.get_word_list(line,wordsList,stopList)
    except FileNotFoundError:
        spamFilelen -= 1
    # 统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
spamDict = wordsDict.copy()

# 测试邮件
#保存预测结果,key为文件名，值为预测类别
print("正在预测测试集结果")
testResult={}

for fileName in tqdm(testFileList):
    testDict.clear( )
    wordsDict.clear()
    wordsList.clear()
    for line in open(fileName, encoding='gb18030'):
        # 过滤掉非中文字符
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("", line)
        # 将每封邮件出现的词保存在wordsList中
        spam.get_word_list(line, wordsList, stopList)

    # 统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
    testDict = wordsDict.copy()

    #通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    wordProbList = spam.getTestWords(testDict, spamDict, normDict, normFilelen, spamFilelen)

    #对每封邮件得到的15个词计算贝叶斯概率  
    try:
        p = spam.calBayes(wordProbList, spamDict, normDict)
    except ZeroDivisionError:
        p = 0.99

    if (p > 0.8):
        testResult.setdefault(fileName, 1) # Spam
    else:
        testResult.setdefault(fileName, 0) # Ham

# 输出result.txt文件
print("正在写入结果文件")
f = open("C:\\Users\\Administrator\\Desktop\\SpamClassfier\\MySpamBayes\\result.txt", 'w')
f.write("TYPE ID\n")
for (fileName, clazz) in tqdm(testResult.items()):
    firstPath, secondPath = fileName[-7:].split("\\")
    if clazz == 1:
        f.write("spam ../Data/" + firstPath + '/' + secondPath + '\n')
    elif clazz == 0:
        f.write("ham ../Data/" + firstPath + '/' + secondPath + '\n')
f.close()