# -*- coding: utf-8 -*-

from numpy import *

def loadSimpData():
    datMat = matrix([
        [1. , 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    分类桩
    :param dataMatrix: 待分类的数据集合
    :param dimen: 维度
    :param threshVal: 分类阀门
    :param threshIneq: 大于阀门还是小于阀门
    :return:
    """
    retArray = ones((shape(dataMatrix)[0], 1))
    # 通过阀门比较对数据进行分类，在阀门一边数据分到-1，另一边数据为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """

    :param dataArr: 输入训练数据
    :param classLabels: 输入训练数据的便签
    :param D: 权值
    :return:
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    # 第一个循环在特征上进行遍历（数据集的列向量）
    for i in range(n):
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 遍历特征列向量的值
        for j in range(-1, int(numSteps) + 1):
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print "split : dim %d, thresh %.2f, thresh ineqal :" \
                      " %s, the weighted error is %.3f" %\
                      (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBootsTrainDs(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D: ", D.T
        #
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ", classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"
        if errorRate == .0: break
    return weakClassArr

def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)

if __name__ == '__main__':
    datArr, labelArr = loadSimpData()
    classifierArr = adaBootsTrainDs(datArr, labelArr, 30)
    adaClassify([0, 0], classifierArr)
