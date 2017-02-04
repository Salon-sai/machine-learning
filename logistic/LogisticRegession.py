# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

def load_data_set():
    data_mat = []; label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def grad_ascent(data_mat_in, class_labels):
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

def stoc_grad_ascent0(data_matrix, class_labels):

    m, n = np.shape(data_matrix)
    alpha = 0.01
    # n维行向量
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights += alpha * error * data_matrix[i]
    return weights

def stoc_grad_ascent1(data_matrix, class_labels, numIter=150):
    m, n = np.shape(data_matrix)
    weight = np.ones(n)
    for i in range(numIter):
        data_index = range(m)
        for j in range(m):
            alpha = 4 / (1.0 + j +i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weight))
            error = class_labels[rand_index] - h
            weight += alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weight


def plot_best_fit(weights, data_mat, label_mat):
    data_arr = np.array(data_mat)
    weights = np.array(weights)
    n = np.shape(data_arr)[0]
    x_cord1 = [];y_cord1 = []
    x_cord2 = [];y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    # calculate the z = W * X
    x = np.arange(-3.0, 3.0, 0.1)
    #  w0 = w1 * x + w2 * y
    y = (-weights[0]-weights[1] * x)/ weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    data_mat, label_mat = load_data_set()
    weights = grad_ascent(data_mat, label_mat)
    plot_best_fit(weights, data_mat, label_mat)

    # data_mat, label_mat = load_data_set()
    # weights = stoc_grad_ascent0(np.array(data_mat), label_mat)
    # plot_best_fit(weights, data_mat, label_mat)

    # data_mat, label_mat = load_data_set()
    # weights = stoc_grad_ascent1(np.array(data_mat), label_mat)
    # plot_best_fit(weights, data_mat, label_mat)



