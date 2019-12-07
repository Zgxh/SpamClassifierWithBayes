'''
����sklearn ��Ҷ˹ģ�����������ʼ����з���
'''
# ������صĿ�
import os
import jieba.analyse as analyse
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import time

start_time = time.time()

# ����ļ�·��
train_path = r'..\data\train\Data' # ѵ�����ļ���·��
test_path = r'..\data\test\Data' # ���Լ��ļ���·��
index_path = r'..\data\index.txt' # 

# �ų�һЩ����Ҫͳ�Ƶĵ���
exclude_words = ['DOC','DOCNO','FROM','TO','SUBJECT','TEXT']

emails = [] # �洢����email�ļ���·��
all_words = [] # ������е������
i = 1

# ��ȡ����ѵ�������ʼ���·��
for f1 in os.listdir(train_path):
    train_dir = os.path.join(train_path, f1)
    emails += [os.path.join(train_dir, f2) for f2 in os.listdir(train_dir)]
    # �����漰�����б��ĺϲ�������emails�洢�����е�ѵ����email��·��

# ��ȡ�ؼ���
for email in emails:
    try:
        f = open(email, encoding='gb18030').read()
    except UnicodeDecodeError:
        print("wrong read:"+email)
    #��ȡ�ؼ��ʣ�����ʣ�
    for key in analyse.extract_tags(f, 50, withWeight=False):
        if key not in exclude_words:
            all_words.append(key)

# ��ȡƵ����ߵ�1500���ʻ�
word_size = 2000
dictionary = Counter(all_words)
dictionary = dictionary.most_common(word_size)

#  ��ȡѵ������ �������� ÿ���ʼ���Ӧ������ʳ��ֵĴ�����Ϊ����

train_matrix = np.zeros((len(emails), word_size))
emailId = 0
#[('DOC', 48288), ('TEXT', 48231) Ӧ��Ҫȥ��һЩ
for email in emails:
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

# ����ѵ�������ı�ǩ����
f = open(index_path)
print(f.readline)#next(f)
train_label = np.zeros(len(emails))
i = 0
for label in train_label:
    s = f.readline()
    if s[:3] == 'ham':
        train_label[i] = 1
    i += 1

# ģ��
model = MultinomialNB()
# ����ѵ��������ѵ����ǩ
np.save(r'../save', train_matrix)
np.save(r'../save', train_label)
# ѵ��ģ��
model.fit(train_matrix, train_label)

# �õ���������
emails.clear()
for f1 in os.listdir(test_path):
    test_dir = os.path.join(test_path, f1)
    emails += [os.path.join(test_dir, f2) for f2 in os.listdir(test_dir)]

test_matrix = np.zeros((len(emails), word_size))
emailId = 0
for email in emails:
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

# Ԥ��
predict = model.predict(test_matrix)
np.save(r'../save', predict)

# ��ȡ���
i = 0
a = open('result.txt', 'w')
a.write("TYPE ID"+"\n")
for email in emails:
    email = "../Data/{0}/{1}".format(email.split("\\")[-2],email.split("\\")[-1])
    if predict[i] == 1:
        tag = 'ham {0}'.format(email)
    else:
        tag = 'spam {0}'.format(email)
    i += 1
    a.write(tag+"\n")

end_time = time.time()
# print("��������ʱ��:{0}".format(end_time -start_time))