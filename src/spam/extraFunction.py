#encoding=utf-8

import jieba
import os

class spamBayes:

    #获得停用词表
    def getStopWords(self):
        stopList = []
        for line in open(r"../src/data/中文停用词表.txt", 'rb'):
            stopList.append(line[:len(line)-1])
    
        return stopList