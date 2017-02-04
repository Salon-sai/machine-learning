# -*- coding: utf-8 -*-
import numpy as np

def load_data_set():
    # 加载训练数据
    posting_list = [['my', 'dog', 'has', 'flea',
                    'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1] # 1 typeof , 0 typeof
    return posting_list, class_vec

# 创建词汇列表，该列表中的词汇不会重复
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 合并成同一个集合当中
    return list(vocabSet)

# 输入参数为词汇表以及某个文档，输出的是文档向量，
# 向量的每一个元素为1或0，分别表示词汇表中的单词在输入文档是否出现
def bag_of_words2_vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print "the word : %s is not in my Vocabulary!" % word
    return returnVec


def train_nb0(train_matrix, train_category):
    """
    :param train_matrix: 一个文档矩阵，存放每个文档中出现的单词
    :param train_category: 每篇文档类别标签所构成的向量，即标记为侮辱性文档还是非侮辱性文档
    :return:
    """
    # 文档总数
    num_train_docs = len(train_matrix)
    # 词典中，单词总数
    num_words = len(train_matrix[0])
    # 在训练样本中，出现侮辱性文档的概率
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    #
    p0_denom = 2.0
    #
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            # 统计每个单词出现侮辱性词汇的次数
            p1_num += train_matrix[i]
            # 统计该句子出现单词的总数
            p1_denom += sum(train_matrix[i])
        else:
            # 统计每个单词出现正常词汇的次数
            p0_num += train_matrix[i]
            # 统计该句子出现单词的总数
            p0_denom += sum(train_matrix[i])

    # 因为概率太小，想相乘时容易出现下溢。采用log函数可以实现：ln(x * y) = ln(x) + ln(y)
    # p(wi|c1) : 在侮辱性句子条件下，出现第i个单词的概率
    p1_vect = np.log(p1_num / p1_denom)
    # p(wi|c0)： 在正常性句子条件下，出现第i个单词的概率
    p0_vect = np.log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2_classify, p0_vec, p1_vec, p_class1):
    """

    :param vec2_classify: 被分类的文档的单词向量
    :param p0_vec: p(w|c0)的条件概率向量：{p(wi|c0)| i = 0...n}
    :param p1_vec: p(w|c1)的条件概率向量：{p(wi|c1)| i = 0...n}
    :param p_class1: 训练样本中出现侮辱性句子的概率
    :return:
    """
    p1 = sum(vec2_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + np.log(1.0 - p_class1)
    return p1 > p0

if __name__ == '__main__':
    listOPosts, listClasses = load_data_set()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for post_in_doc in listOPosts:
        trainMat.append(bag_of_words2_vec(myVocabList, post_in_doc))
    p0_vect, p1_vect, p_abusive = train_nb0(trainMat, listClasses)
    testEntry = ['I', 'love', 'my', 'stupid']
    this_doc = bag_of_words2_vec(myVocabList, testEntry)
    print testEntry, ' classified as : ', classify_nb(this_doc, p0_vect, p1_vect, p_abusive)