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

if __name__ == '__main__':
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