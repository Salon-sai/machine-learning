# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 求解行列式的值是否等于0，若等于0说明矩阵不可逆
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 局部加权值线性回归
# (X.T * w * T.I) * X.T * W * y
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        # 构建对角矩阵weights,点x与x(i)越近,w(i,i)将会越大
        # 随着样本点于待测试点距离的递增,weight的值不断衰减
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def drawLwlr():
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    sortIndex = xMat[:, 1].argsort(0)
    xSort = xMat[sortIndex][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[sortIndex])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()


def drawStandRegres():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)

    xMat = mat(xArr)
    yMat = mat(yArr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 描画训练数据散点图
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    # 需要将训练数据掉转 y = kx + b, 其中k,b是yHat中的内容
    # x在xMat中的第1列， 1在xMat中的第2列。
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

if __name__ == '__main__':
    drawLwlr()