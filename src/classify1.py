import os
import jieba.analyse as analyse
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm 


# 文件路径
train_path = r'.\data\train\Data' # 训练集文件夹路径
test_path = r'.\data\test\Data' # 测试集文件夹路径
index_path = r'.\data\index.txt' # 

# 排除一些不必要统计的单词
exclude_words = ['DOC','DOCNO','FROM','TO','SUBJECT','TEXT']

emails = [] # 存储所有email文件的路径
all_words = [] # 存错所有的主题词
i = 1

# 提取所有训练集里邮件的路径
for f1 in os.listdir(train_path):
    train_dir = os.path.join(train_path, f1)
    emails += [os.path.join(train_dir, f2) for f2 in os.listdir(train_dir)]
    # 这里涉及到了列表的合并操作。emails存储了所有的训练集email的路径

# 提取关键词
for email in tqdm(emails):
    try:
        f = open(email, encoding='gb18030').read()
    except UnicodeDecodeError:
        print("wrong read:"+email)
    #提取关键词（主题词）
    for key in analyse.extract_tags(f, 50, withWeight=False):
        if key not in exclude_words:
            all_words.append(key)

# 提取频率最高的1500个词汇
word_size = 1200
dictionary = Counter(all_words)
dictionary = dictionary.most_common(word_size)

#  提取训练矩阵 特征矩阵 每个邮件对应的主题词出现的次数作为特征

train_matrix = np.zeros((len(emails), word_size))
emailId = 0
#[('DOC', 48288), ('TEXT', 48231) 应该要去掉一些
for email in tqdm(emails):
    try:
        f = open(email, encoding='gb18030').read()
    except UnicodeDecodeError:
        print("wrong read:"+email)
    keys = analyse.extract_tags(f, 50, withWeight=False)
    for key in keys:
        wordId = 0
        for index, word in enumerate(dictionary):
            if word[0] == key:
                wordId = index
                train_matrix[emailId][wordId] = keys.count(key)
    emailId += 1 

# 制作训练样本的标签集合
f = open(index_path)
print(f.readline)#next(f)
train_label = np.zeros(len(emails))
i = 0
for label in train_label:
    s = f.readline()
    if s[:3] == 'ham':
        train_label[i] = 1
    i += 1

# 模型
model = MultinomialNB()
# 保存训练特征和训练标签
np.save(r'C:\Users\Administrator\Desktop\SpamClassfier\MySpamBayes\classify1.py\train_matrix', train_matrix)
np.save(r'C:\Users\Administrator\Desktop\SpamClassfier\MySpamBayes\classify1.py\train_label', train_label)
# 训练模型
model.fit(train_matrix, train_label)

# 得到测试数据
emails.clear()
for f1 in os.listdir(test_path):
    test_dir = os.path.join(test_path, f1)
    emails += [os.path.join(test_dir, f2) for f2 in os.listdir(test_dir)]

test_matrix = np.zeros((len(emails), word_size))
emailId = 0
for email in tqdm(emails):
    try:
        f = open(email, encoding='gb18030').read()
    except UnicodeDecodeError:
        print("wrong email:" + email)
    
    keys = analyse.extract_tags(f, 50, withWeight = False)
    for key in keys:
        wordId = 0
        for index, word in enumerate(dictionary):
            if key == word[0]:
                wordId = index
                test_matrix[emailId][wordId] = keys.count(key)
    emailId += 1

# 预测
predict = model.predict(test_matrix)
np.save(r'C:\Users\Administrator\Desktop\SpamClassfier\MySpamBayes', predict)

# 获取结果
i = 0
a = open('index.txt', 'w')
a.write("TYPE ID"+"\n")
for email in emails:
    email = "../Data/{0}/{1}".format(email.split("\\")[-2],email.split("\\")[-1])
    if predict[i] == 1:
        tag = 'ham {0}'.format(email)
    else:
        tag = 'spam {0}'.format(email)
    i += 1
    a.write(tag+"\n")

