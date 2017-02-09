# -*- coding: utf-8 -*-

from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    # 获取特征数据的维数
    numFeatures = len(dataSet[0]) -1
    # 计算总的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最佳最佳的香农熵和最好分类特征的位置(即index)
    bestInfoGain = 0.0; bestFeature = -1
    # 循环所有特征
    for i in range(numFeatures):
        # 获取数据集中的所有第i个特征的值(构成列向量)
        featList = [example[i] for example in dataSet]
        # 该特征列向量的值(唯一)
        uniqueVals = set(featList)
        # 定义新的香农熵值
        newEntropy = 0.0
        # 计算按照第i列特征分类后的总香农熵值
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息熵
        infoGain = baseEntropy - newEntropy
        # 判断当前信息熵是否最大，若最大该特征为最佳分类特征
        if (infoGain > bestInfoGain):
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同的停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选取集合中最佳的特征作为分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 拷贝一份列表到下一个调用当中，避免影响其他createTree
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    myData, labels = createDataSet()
    print createTree(myData, labels)
